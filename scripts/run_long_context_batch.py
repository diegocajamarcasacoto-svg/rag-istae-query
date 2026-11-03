"""Corre un lote de preguntas en las variantes de contexto largo y resume los resultados.

Uso:
    python scripts/run_long_context_batch.py evaluation/cases/pruebas_humo.json
    python scripts/run_long_context_batch.py casos.json --use-cache --tpm-limit 250000

Cada caso del archivo JSON tiene:
    id, question, variant ("A" o "AJ"), documents (lista o null = corpus completo),
    expected_units (IDs de unidades que una respuesta correcta debería citar; [] si no hay respuesta),
    answerable (false para preguntas sin respuesta en el corpus).

El script espera cuando la próxima consulta superaría el límite de tokens por
minuto (--tpm-limit) y reintenta una vez ante un error 429.
Guarda en la carpeta de resultados: lote_<fecha>.json (todo) y lote_<fecha>.md (resumen).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from pathlib import Path

from _long_context_cli import DEFAULT_SETTINGS, build_variants, run, save_results
from query.long_context.variant import Answer


class TokenThrottle:
    """Ventana deslizante de 60 s para no superar un límite de tokens por minuto."""

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.events: deque[tuple[float, int]] = deque()

    def wait_for(self, tokens: int) -> float:
        if not self.limit:
            return 0.0
        if tokens > self.limit:
            raise ValueError(
                f"Una sola consulta (~{tokens:,} tokens) supera el límite de {self.limit:,} tokens por minuto. "
                "Reduzca los documentos o use un plan con límites mayores."
            )
        waited = 0.0
        while True:
            now = time.monotonic()
            while self.events and now - self.events[0][0] >= 60:
                self.events.popleft()
            used = sum(count for _, count in self.events)
            if used + tokens <= self.limit:
                return waited
            pause = 60 - (now - self.events[0][0]) + 1
            time.sleep(pause)
            waited += pause

    def record(self, tokens: int) -> None:
        self.events.append((time.monotonic(), tokens))


def _retry_delay(exc: Exception) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(match.group(1)) + 2 if match else 35.0


def _answer_with_retry(variant, question: str):
    try:
        from google.genai import errors as genai_errors
    except ImportError:  # pragma: no cover
        genai_errors = None
    try:
        return variant.answer(question)
    except Exception as exc:  # noqa: BLE001
        if (genai_errors is not None and isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429
                and "CachedContentStorage" not in str(exc)):  # el límite de caché no se resuelve esperando
            delay = _retry_delay(exc)
            print(f"    429: esperando {delay:.0f} s y reintentando una vez…")
            time.sleep(delay)
            return variant.answer(question)
        raise


def evaluate(answer: Answer, case: dict) -> dict:
    cited = {c.unit_id for c in answer.citations}
    expected = set(case.get("expected_units", []))
    known = [c for c in answer.citations if c.known]
    checked = [c for c in answer.citations if c.quote_verified is not None]
    answerable = case.get("answerable", True)
    return {
        "expected_recall": (len(cited & expected) / len(expected)) if expected else None,
        "expected_missing": sorted(expected - cited),
        "valid_citation_ratio": (len(known) / len(answer.citations)) if answer.citations else None,
        "verified_quote_ratio": (sum(1 for c in checked if c.quote_verified) / len(checked)) if checked else None,
        "invented_ids": answer.unknown_citations,
        "abstained": answer.abstained,
        "abstention_correct": (answer.abstained == (not answerable)),
    }


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Sí" if value else "No"
    if isinstance(value, float):
        return f"{value:.0%}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", type=Path)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--use-cache", action="store_true", help="Activa la caché explícita para este lote.")
    parser.add_argument("--tpm-limit", type=int, default=250_000, help="Tokens de entrada por minuto (0 = sin límite).")
    args = parser.parse_args(argv)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    throttle = TokenThrottle(args.tpm_limit or None)
    overrides = {"use_cache": True} if args.use_cache else None
    results = []
    settings = None
    # Las preguntas con la misma variante y el mismo corpus comparten cliente y
    # variante: así reutilizan la caché, que se elimina recién al final del lote.
    built: dict[tuple, tuple] = {}
    try:
        for number, case in enumerate(cases, 1):
            documents = set(case["documents"]) if case.get("documents") else None
            key = (case["variant"], frozenset(documents or ()))
            if key not in built:
                settings, client, variants = build_variants(args.settings, [case["variant"]], documents, overrides)
                built[key] = (client, variants[case["variant"]])
            client, variant = built[key]
            estimate = int(len(variant.context) / 4.4) + 3000
            waited = throttle.wait_for(estimate)
            print(f"[{number}/{len(cases)}] {case['id']} · {case['variant']} · {', '.join(sorted(documents)) if documents else 'corpus completo'}"
                  + (f" (esperó {waited:.0f} s)" if waited else ""))
            answer = _answer_with_retry(variant, case["question"])
            _process(answer, case, results, throttle, estimate)
    finally:
        for client, _ in built.values():
            close = getattr(client, "close", None)
            if callable(close):
                close()
    return _report(results, settings, args)


def _process(answer: Answer, case: dict, results: list, throttle: TokenThrottle, estimate: int) -> None:
    throttle.record(answer.input_tokens or estimate)
    metrics = evaluate(answer, case)
    print(f"    citas esperadas encontradas: {_fmt(metrics['expected_recall'])} · citas válidas: {_fmt(metrics['valid_citation_ratio'])}"
          f" · citas textuales verificadas: {_fmt(metrics['verified_quote_ratio'])} · abstención correcta: {_fmt(metrics['abstention_correct'])}"
          f" · tokens en caché: {answer.cached_input_tokens} · {answer.latency_ms / 1000:.1f} s · fin: {answer.finish_reason}")
    results.append({"case": case, "metrics": metrics, "answer": answer.to_dict()})


def _report(results: list, settings, args) -> int:
    lines = [
        "# Lote de pruebas · contexto largo",
        "",
        f"Modelo: {settings.model if settings else '—'} · caché explícita: {'sí' if args.use_cache else 'no'} · casos: {len(results)}",
        "",
        "| Caso | Variante | Documentos | Citas esperadas encontradas | Citas válidas | Citas textuales verificadas | IDs inventados | Abstención correcta | Tokens entrada (caché) | Razonamiento | Latencia | Fin |",
        "|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for item in results:
        case, m, a = item["case"], item["metrics"], item["answer"]
        docs = ", ".join(case["documents"]) if case.get("documents") else "completo"
        lines.append(
            f"| {case['id']} | {case['variant']} | {docs} | {_fmt(m['expected_recall'])} | {_fmt(m['valid_citation_ratio'])} | "
            f"{_fmt(m['verified_quote_ratio'])} | {', '.join(m['invented_ids']) or '—'} | {_fmt(m['abstention_correct'])} | "
            f"{a['input_tokens']} ({a['cached_input_tokens']}) | {a['extra'].get('thoughts_tokens')} | {a['latency_ms'] / 1000:.1f} s | {a['finish_reason']} |"
        )
    lines.append("")
    for item in results:
        case, a, m = item["case"], item["answer"], item["metrics"]
        lines += [f"## {case['id']} · {case['variant']}", "", f"**Pregunta:** {case['question']}", ""]
        if m["expected_missing"]:
            lines += [f"Citas esperadas no encontradas: {', '.join(m['expected_missing'])}", ""]
        lines += [a["text"], ""]
    json_path, md_path = save_results(settings.results_dir, "lote", {"cases": results}, "\n".join(lines) + "\n")
    print(f"\nResultados guardados en:\n  {md_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))

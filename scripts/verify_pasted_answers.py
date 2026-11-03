"""Verifica las citas de respuestas copiadas desde AI Studio (ejecución manual).

Guarde cada respuesta (solo la respuesta final, sin el resumen del razonamiento)
como archivo de texto en la carpeta de respuestas, con este nombre:
    <VARIANTE>_<PREGUNTA>.txt      por ejemplo  A_Q01.txt   AJ_Q27.txt   A_SIN-RESPUESTA.txt

Uso:
    python scripts/verify_pasted_answers.py
    python scripts/verify_pasted_answers.py --answers-dir "<carpeta>" --cases evaluation/cases/pruebas_humo.json

Las citas esperadas se toman del archivo de casos (el ID del caso sin el sufijo
de variante, p. ej. "Q27-AJ" → "Q27"). Genera verificacion_<fecha>.md y .json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _long_context_cli import DEFAULT_SETTINGS, ROOT, run, save_results
from query.long_context.citations import parse_answer
from query.long_context.settings import load_settings


def _load_units(outputs_dir: Path) -> dict[str, tuple[str, str]]:
    units = {}
    with (outputs_dir / "long_context" / "unidades.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            units[row["unit_id"]] = (row["document_id"], row["text"])
    return units


def _load_cases(path: Path) -> dict[str, dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    by_question: dict[str, dict] = {}
    for case in cases:
        base = case["id"].upper()
        for suffix in ("-AJ", "-A"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        by_question.setdefault(base, case)
    return by_question


def _pct(value) -> str:
    return "—" if value is None else f"{value:.0%}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--answers-dir", type=Path, help="Por defecto: <outputs>/long_context/para_ai_studio/respuestas")
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation" / "cases" / "pruebas_humo.json")
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    answers_dir = args.answers_dir or settings.outputs_dir / "long_context" / "para_ai_studio" / "respuestas"
    files = sorted(answers_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No hay respuestas .txt en {answers_dir}")
    units = _load_units(settings.outputs_dir)
    cases = _load_cases(args.cases) if args.cases.exists() else {}

    rows, details = [], []
    for path in files:
        variant, _, question_id = path.stem.partition("_")
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            print(f"{path.name}: vacío, se omite (pegue la respuesta y vuelva a ejecutar)")
            continue
        parsed = parse_answer(text, units)
        case = cases.get(question_id.upper(), {})
        expected = set(case.get("expected_units", []))
        cited = {c.unit_id for c in parsed.citations}
        answerable = case.get("answerable", True)
        result = {
            "file": path.name,
            "variant": variant,
            "question_id": question_id,
            "expected_recall": (len(cited & expected) / len(expected)) if expected else None,
            "expected_missing": sorted(expected - cited),
            "citations": [c.to_dict() for c in parsed.citations],
            "invented_ids": parsed.unknown_ids,
            "verified_quote_ratio": parsed.verified_quote_ratio,
            "abstained": parsed.abstained,
            "abstention_correct": (parsed.abstained == (not answerable)) if case else None,
        }
        details.append(result)
        rows.append(
            f"| {path.name} | {variant} | {_pct(result['expected_recall'])} | {len(parsed.citations)} | "
            f"{', '.join(parsed.unknown_ids) or '—'} | {_pct(parsed.verified_quote_ratio)} | "
            f"{'—' if result['abstention_correct'] is None else ('Sí' if result['abstention_correct'] else 'No')} |"
        )
        print(f"{path.name}: citas={len(parsed.citations)} esperadas={_pct(result['expected_recall'])} "
              f"inventadas={parsed.unknown_ids or 'ninguna'} textuales verificadas={_pct(parsed.verified_quote_ratio)}")
        for c in parsed.citations:
            if not c.known or c.quote_verified is False:
                print(f"    revisar [{c.unit_id}]: {'no existe en el corpus' if not c.known else 'cita textual no encontrada'}")

    lines = [
        "# Verificación de respuestas de AI Studio",
        "",
        f"Carpeta: {answers_dir}",
        "",
        "| Archivo | Variante | Citas esperadas encontradas | Citas | IDs inventados | Citas textuales verificadas | Abstención correcta |",
        "|---|---|---:|---:|---|---:|---|",
        *rows,
    ]
    json_path, md_path = save_results(settings.results_dir, "verificacion_ai_studio", {"answers": details}, "\n".join(lines) + "\n")
    print(f"\nResultados guardados en:\n  {md_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))

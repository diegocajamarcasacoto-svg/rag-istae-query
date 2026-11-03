"""Hace una pregunta a la variante de contexto largo (A o A+J) y verifica las citas.

Uso:
    python scripts/ask_long_context.py "¿Qué necesito para que me declaren egresado?" --variant A --documents INT08,EXT01
    python scripts/ask_long_context.py "..." --variant AJ --stream --json-output resultado.json

Con --documents se usa un corpus reducido (recomendado en el plan gratuito).
Cada ejecución consume tokens del proveedor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _long_context_cli import DEFAULT_SETTINGS, build_variants, parse_documents, run, save_results
from query.long_context.variant import Answer


def _print_answer(answer: Answer) -> None:
    print("\n=== VERIFICACIÓN DE CITAS ===")
    if not answer.citations:
        print("Sin citas." + (" (El modelo se abstuvo.)" if answer.abstained else ""))
    for citation in answer.citations:
        status = "existe" if citation.known else "NO EXISTE EN EL CORPUS"
        quote = {True: "cita textual verificada", False: "cita textual NO encontrada", None: "sin cita textual"}[citation.quote_verified]
        print(f"- [{citation.unit_id}] {status}; {quote}")
    if answer.cited_without_source:
        print(f"Citados sin cita textual en 'Fuentes': {', '.join(answer.cited_without_source)}")
    print("\n=== USO ===")
    print(f"Variante: {answer.variant_name} · modelo: {answer.model} · versión de corpus: {answer.corpus_version}")
    print(f"Tokens: entrada={answer.input_tokens} (en caché={answer.cached_input_tokens}) · salida={answer.output_tokens}")
    print(f"Latencia: {answer.latency_ms / 1000:.1f} s · motivo de fin: {answer.finish_reason}")
    if answer.extra.get("model_mismatch"):
        print(f"AVISO: se pidió {answer.extra.get('requested_model')} pero respondió {answer.model}. "
              "El modelo puede estar deprecado y redirigido; no mezcle estos resultados con los de otro modelo.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--variant", default="A", choices=["A", "AJ"])
    parser.add_argument("--documents", help="Lista de document_id separada por comas (corpus reducido).")
    parser.add_argument("--stream", action="store_true", help="Muestra la respuesta a medida que llega.")
    parser.add_argument("--json-output", type=Path, help="Ruta alternativa (por defecto: carpeta de resultados con fecha y hora).")
    args = parser.parse_args(argv)

    settings, client, variants = build_variants(args.settings, [args.variant], parse_documents(args.documents))
    variant = variants[args.variant]
    try:
        print("=== RESPUESTA ===")
        if args.stream:
            answer = None
            for event in variant.stream(args.question):
                if isinstance(event, Answer):
                    answer = event
                else:
                    print(event, end="", flush=True)
            print()
        else:
            answer = variant.answer(args.question)
            print(answer.text)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    _print_answer(answer)
    json_path, md_path = save_results(
        settings.results_dir, f"respuesta_{args.variant}", answer.to_dict(), _markdown(answer, args.documents), args.json_output
    )
    print(f"\nResultados guardados en:\n  {md_path}\n  {json_path}")
    return 0


def _markdown(answer: Answer, documents: str | None) -> str:
    rows = []
    for c in answer.citations:
        quote = {True: "verificada", False: "NO encontrada", None: "sin cita textual"}[c.quote_verified]
        rows.append(f"| [{c.unit_id}] | {'Sí' if c.known else 'No'} | {quote} |")
    lines = [
        f"# Respuesta · {answer.variant_name}",
        "",
        f"**Pregunta:** {answer.question}",
        "",
        f"Modelo: {answer.model} · corpus: {documents or 'completo'} · versión de corpus: {answer.corpus_version}",
        "",
        "## Respuesta",
        "",
        answer.text,
        "",
        "## Verificación de citas",
        "",
        "| Cita | ¿Existe en el corpus? | Cita textual |",
        "|---|---|---|",
        *(rows or ["| — | — | — |"]),
        "",
        f"Abstención: {'Sí' if answer.abstained else 'No'} · citas sin cita textual: {', '.join(answer.cited_without_source) or 'ninguna'}",
        "",
        "## Uso",
        "",
        f"- Tokens de entrada: {answer.input_tokens} (en caché: {answer.cached_input_tokens})",
        f"- Tokens de salida: {answer.output_tokens}",
        f"- Latencia: {answer.latency_ms / 1000:.1f} s · motivo de fin: {answer.finish_reason}",
    ]
    if answer.extra.get("model_mismatch"):
        lines.append(f"- AVISO: se pidió {answer.extra.get('requested_model')} y respondió {answer.model}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(run(main))

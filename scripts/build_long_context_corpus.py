"""Construye los corpus de contexto largo (A y A+J) desde la salida de la ingesta.

Uso:
    python scripts/build_long_context_corpus.py --outputs "<storage>/outputs" --out-dir "<storage>/outputs/long_context"

Genera en --out-dir:
    corpus_A.md        documentos completos en orden neutro (variante A)
    corpus_AJ.md       documentos por rango jerárquico + relaciones (variante A+J)
    unidades.jsonl     una línea por unidad citable (artículo, sección, disposición)
    informe.json       tamaños, estructura detectada, limpieza y advertencias
    informe.md         resumen legible del informe
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.long_context.catalog import load_catalog  # noqa: E402
from query.long_context.corpus import build_report, load_corpus, render_corpus, units_records  # noqa: E402


def _markdown_report(report: dict, catalog_meta: dict) -> str:
    lines = ["# Informe del corpus de contexto largo", ""]
    lines.append(f"- Documentos: {report['documents']}")
    lines.append(f"- Unidades citables: {report['units']}")
    lines.append(f"- Caracteres antes / después de limpiar: {report['raw_chars']:,} / {report['clean_chars']:,}")
    lines.append(f"- Rangos jerárquicos: {catalog_meta.get('rank_status', 'sin catálogo')}")
    lines.append("")
    lines.append("| Corpus | Caracteres | Tokens estimados (÷4 – ÷3,5) |")
    lines.append("|---|---:|---:|")
    for name, stats in report["corpora"].items():
        lines.append(
            f"| {name} | {stats['chars']:,} | {stats['tokens_estimate_4_chars']:,} – {stats['tokens_estimate_3_5_chars']:,} |"
        )
    lines.append("")
    lines.append("Los tokens son una estimación; el conteo real se hace con el tokenizador del modelo.")
    lines.append("")
    lines.append("| Documento | Rango | Modo | Artículos | Secciones | Disposiciones | Caracteres | Advertencias |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|")
    for doc in report["per_document"]:
        warnings = "; ".join(doc["warnings"]) or "—"
        lines.append(
            f"| {doc['document_id']} {doc['name'][:45]} | {doc['rank']} | {doc['mode']} | {doc['articles']} | "
            f"{doc['sections']} | {doc['dispositions']} | {doc['clean_chars']:,} | {warnings} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", type=Path, required=True, help="Carpeta outputs de la ingesta.")
    parser.add_argument("--catalog", type=Path, default=ROOT / "config" / "catalogo_normativo.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    if args.catalog.exists():
        catalog, relations, meta = load_catalog(args.catalog)
    else:
        print(f"Aviso: no existe {args.catalog}; se construye sin rangos ni relaciones.")
        catalog, relations, meta = {}, [], {}

    documents = load_corpus(args.outputs, catalog)
    corpora = {
        "A": render_corpus(documents, "A"),
        "AJ": render_corpus(documents, "AJ", relations, meta.get("rank_labels", {})),
    }
    report = build_report(documents, corpora)
    report["catalog"] = {key: meta.get(key) for key in ("source", "source_sha256", "rank_status")}
    report["duration_seconds"] = round(time.perf_counter() - started, 3)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in corpora.items():
        (args.out_dir / f"corpus_{name}.md").write_text(text, encoding="utf-8")
    with (args.out_dir / "unidades.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in units_records(documents):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.out_dir / "informe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "informe.md").write_text(_markdown_report(report, meta), encoding="utf-8")

    for name, stats in report["corpora"].items():
        print(f"corpus_{name}.md: {stats['chars']:,} caracteres ≈ {stats['tokens_estimate_4_chars']:,}–{stats['tokens_estimate_3_5_chars']:,} tokens (estimado)")
    print(f"Unidades citables: {report['units']:,} en {report['documents']} documentos")
    print(f"Resultados en: {args.out_dir} ({report['duration_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

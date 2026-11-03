"""Genera config/catalogo_normativo.json a partir de la Matriz de reglamentos.

Uso:
    python scripts/build_catalog_from_matriz.py --matriz "<ruta>/Matriz_Reglamentos_TESIS_ISTAE_v5.xlsx"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.long_context.catalog import build_catalog_from_matriz  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matriz", type=Path, required=True, help="Archivo Excel de la Matriz de reglamentos.")
    parser.add_argument("--out", type=Path, default=ROOT / "config" / "catalogo_normativo.json")
    args = parser.parse_args(argv)

    catalog = build_catalog_from_matriz(args.matriz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    linked = sum(1 for rel in catalog["relations"] if rel["source_document_id"] and rel["target_document_id"])
    print(f"Documentos: {len(catalog['documents'])}")
    print(f"Relaciones: {len(catalog['relations'])} (con ambos extremos en el corpus: {linked})")
    by_rank: dict[int, int] = {}
    for doc in catalog["documents"]:
        by_rank[doc["rank"]] = by_rank.get(doc["rank"], 0) + 1
    for rank in sorted(by_rank):
        print(f"  Rango {rank} ({catalog['rank_labels'][str(rank)]}): {by_rank[rank]}")
    print(f"Estado de los rangos: {catalog['rank_status']}")
    print(f"Catálogo escrito en: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

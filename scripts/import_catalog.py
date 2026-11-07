"""Importa ``catalogo_normativo.json`` a ``storage/catalog`` verificando su procedencia.

El catálogo lo genera ``rag-istae-api`` desde ``Matriz_Reglamentos_TESIS_ISTAE_v5.xlsx``.
Este script solo **lee** el archivo de origen (nunca escribe en la API), comprueba
que su ``source_sha256`` coincida con la matriz de ``storage/corpus`` y que el
catálogo sea válido, y lo copia sin modificarlo.

Uso:
    python .\\scripts\\import_catalog.py --source <ruta a rag-istae-api>\\data\\catalogo_normativo.json \\
        --matrix <storage>\\corpus\\Matriz_Reglamentos_TESIS_ISTAE_v5.xlsx \\
        --dest <storage>\\catalog\\catalogo_normativo.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.hierarchy.catalog import catalog_from_dict, file_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="catalogo_normativo.json de origen.")
    parser.add_argument("--matrix", type=Path, required=True, help="Matriz .xlsx de la que se generó el catálogo.")
    parser.add_argument("--dest", type=Path, required=True, help="Destino en storage/catalog.")
    parser.add_argument("--force", action="store_true", help="Sobrescribe el destino si existe y difiere.")
    args = parser.parse_args(argv)

    data = json.loads(args.source.read_text(encoding="utf-8"))
    catalog = catalog_from_dict(data)
    matrix_sha = file_sha256(args.matrix)
    if catalog.source_sha256 != matrix_sha:
        print(
            f"ERROR: source_sha256 del catálogo ({catalog.source_sha256}) no coincide con la matriz "
            f"{args.matrix.name} ({matrix_sha}).",
            file=sys.stderr,
        )
        return 1

    if args.dest.exists():
        if file_sha256(args.dest) == file_sha256(args.source):
            print(f"El destino ya está actualizado: {args.dest}")
            return 0
        if not args.force:
            print(f"ERROR: {args.dest} existe y difiere; use --force para reemplazarlo.", file=sys.stderr)
            return 1
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source, args.dest)
    ranks = sorted({doc.rank for doc in catalog.documents.values()})
    print(f"Catálogo importado: {args.dest}")
    print(f"  normas={len(catalog.documents)} relaciones={len(catalog.relations)} rangos={ranks}")
    print(f"  matriz verificada: {args.matrix.name} sha256={matrix_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

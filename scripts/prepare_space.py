"""Arma la carpeta del Space de Hugging Face para el servicio /retrieve.

Junta en ``--out``:
- ``Dockerfile`` y ``requirements-space.txt`` (de ``deploy/huggingface/``);
- ``README.md``: la configuración del Space (``sdk: docker``, ``app_port: 7860``),
  con la versión del paquete de datos;
- ``src/query/``: el paquete del servicio (sin ``__pycache__``);
- ``data/``: el paquete de datos generado por ``rag-istae-resources/scripts/build_data_bundle.py``
  (modelos, Qdrant, ``units.jsonl``, catálogo y ``MANIFEST.json``), verificado contra su manifiesto.

Esa carpeta se sube tal cual al repositorio del Space (ver docs/guia_publicacion_huggingface.md)
y también sirve para probar la imagen localmente con Docker.

Uso:
    python .\\scripts\\prepare_space.py --data E:\\rag-istae-paquete\\storage --out E:\\rag-istae-space [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy" / "huggingface"
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(data: Path) -> dict:
    manifest_path = data / "MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit(f"No existe {manifest_path}: --data debe ser la carpeta storage del paquete de datos.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    print(f"Verificando {len(manifest['files'])} archivos del paquete {manifest['version']}…")
    for entry in manifest["files"]:
        path = data / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise SystemExit(f"El paquete no coincide con su manifiesto: {entry['path']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Arma la carpeta del Space de Hugging Face del servicio /retrieve.")
    parser.add_argument("--data", type=Path, required=True, help="Carpeta storage del paquete de datos (con MANIFEST.json).")
    parser.add_argument("--out", type=Path, required=True, help="Carpeta de salida (repositorio del Space).")
    parser.add_argument("--force", action="store_true", help="Reemplaza --out si ya existe.")
    args = parser.parse_args()

    data = args.data.resolve()
    out = args.out.resolve()
    if ROOT in out.parents or out == ROOT:
        raise SystemExit("--out no puede estar dentro del repositorio rag-istae-query.")
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} ya existe; use --force para reemplazarlo.")
        shutil.rmtree(out)

    manifest = verify_bundle(data)
    out.mkdir(parents=True)
    for name in ("Dockerfile", "requirements-space.txt"):
        shutil.copy2(DEPLOY / name, out / name)
    readme = (DEPLOY / "README.space.md").read_text(encoding="utf-8")
    readme = readme.replace("{bundle_version}", manifest["version"]).replace(
        "{units_sha256}", manifest["index_version"]["units_sha256"]
    )
    (out / "README.md").write_text(readme, encoding="utf-8")
    shutil.copytree(ROOT / "src" / "query", out / "src" / "query",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"Copiando datos a {out / 'data'}…")
    shutil.copytree(data, out / "data")

    total = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    print(f"Space listo en {out} ({total / 1e9:.2f} GB).")
    print(f"Paquete de datos {manifest['version']}: units_sha256 = {manifest['index_version']['units_sha256']}")


if __name__ == "__main__":
    sys.exit(main())

"""Descarga el paquete de datos del servicio /retrieve desde un repositorio de dataset de Hugging Face y lo verifica.

Se ejecuta al construir la imagen del Space (ver Dockerfile). El repositorio del Space no puede guardar
más de 1 GB, y el paquete LB1 pesa 2,36 GB (modelos, Qdrant, units.jsonl, catálogo), así que los datos
viven en un repositorio de dataset privado y se descargan fijados a un commit.

Verificaciones, en orden (cualquier diferencia detiene la construcción):
1. el sha256 de ``MANIFEST.json`` coincide con el esperado (fijado por ``prepare_space.py``);
2. el sha256 de cada archivo listado en el manifiesto coincide con el descargado.

El token se lee de la variable de entorno ``HF_TOKEN``; nunca se imprime.

    HF_TOKEN=… python fetch_data.py --repo usuario/rag-istae-data-lb1 --revision <commit> \
        --manifest-sha256 <sha256> --out /app/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CHUNK = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(data: Path, manifest_sha256: str) -> dict:
    """Verifica el manifiesto y cada archivo listado. Devuelve el manifiesto."""
    manifest_path = data / "MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Falta {manifest_path}.")
    actual = sha256(manifest_path)
    if actual != manifest_sha256:
        raise SystemExit(f"MANIFEST.json no coincide: se esperaba {manifest_sha256}, se obtuvo {actual}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for entry in manifest["files"]:
        path = data / entry["path"]
        if not path.is_file():
            raise SystemExit(f"Falta un archivo del paquete: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise SystemExit(f"El archivo no coincide con el manifiesto: {entry['path']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga y verifica el paquete de datos del servicio /retrieve.")
    parser.add_argument("--repo", required=True, help="Repositorio de dataset, p. ej. usuario/rag-istae-data-lb1.")
    parser.add_argument("--revision", required=True, help="Commit del repositorio de datos (fijo, no una rama).")
    parser.add_argument("--manifest-sha256", required=True, help="sha256 esperado de MANIFEST.json.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        raise SystemExit("--revision debe ser el hash completo de un commit (40 caracteres), no una rama.")
    token = os.environ.get("HF_TOKEN") or None
    if token is None:
        raise SystemExit("Defina el secret HF_TOKEN del Space (lectura sobre el repositorio de datos).")

    from huggingface_hub import snapshot_download

    print(f"Descargando {args.repo}@{args.revision[:12]} en {args.out}…", flush=True)
    snapshot_download(repo_id=args.repo, repo_type="dataset", revision=args.revision,
                      local_dir=args.out, token=token)
    # La descarga deja metadatos en .cache/huggingface dentro de --out; no forman parte del paquete.
    manifest = verify(args.out, args.manifest_sha256)
    total = sum(entry.get("bytes", 0) for entry in manifest["files"])
    print(f"Paquete {manifest['version']} verificado: {len(manifest['files'])} archivos, {total / 1e9:.2f} GB, "
          f"units_sha256 = {manifest['index_version']['units_sha256']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

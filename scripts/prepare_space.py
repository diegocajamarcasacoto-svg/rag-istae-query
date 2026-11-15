"""Arma la carpeta del Space de Hugging Face para el servicio /retrieve.

El repositorio de un Space admite como máximo 1 GB y el paquete de datos LB1 pesa 2,36 GB, así que los
datos viven en un repositorio de dataset PRIVADO (``--data-repo``) y la imagen los descarga al construirse,
fijados a un commit y verificados con sha256 (``deploy/huggingface/fetch_data.py``).

Junta en ``--out`` (unos cientos de KB):
- ``Dockerfile``, con el repositorio de datos, el commit y el sha256 de ``MANIFEST.json`` ya fijados;
- ``requirements-space.txt`` y ``fetch_data.py`` (de ``deploy/huggingface/``);
- ``README.md``: la configuración del Space (``sdk: docker``, ``app_port: 7860``, ``license: other``);
- ``src/query/``: el paquete del servicio (sin ``__pycache__``).

Antes de escribir, verifica el paquete local (``--data``) contra su manifiesto y comprueba que el
repositorio de datos, en ese commit, tenga el mismo ``MANIFEST.json`` y los mismos archivos (tamaño y, en
los archivos grandes, sha256). Usa la sesión de ``hf auth login``; no lee ni imprime el token.

Uso:
    python .\\scripts\\prepare_space.py --data E:\\rag-istae-paquete\\storage \\
        --data-repo usuario/rag-istae-data-lb1 [--data-revision <commit>] --out E:\\rag-istae-space [--force]

Sin ``--data-revision`` se fija el commit actual de la rama principal del repositorio de datos.
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
    print(f"Verificando {len(manifest['files'])} archivos del paquete local {manifest['version']}…")
    for entry in manifest["files"]:
        path = data / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise SystemExit(f"El paquete no coincide con su manifiesto: {entry['path']}")
    return manifest


def verify_remote(repo: str, revision: str | None, manifest: dict, manifest_sha256: str) -> str:
    """Comprueba que el repositorio de datos tenga el mismo paquete. Devuelve el commit fijado."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(repo, revision=revision, files_metadata=True)
    commit = info.sha
    print(f"Repositorio de datos {repo} @ {commit} (privado: {info.private})")
    if not info.private:
        raise SystemExit(f"{repo} es público: el paquete incluye normativa interna y debe ser privado.")
    remote = {s.rfilename: s for s in info.siblings or []}
    for entry in manifest["files"]:
        item = remote.get(entry["path"])
        if item is None:
            raise SystemExit(f"Falta en {repo}@{commit[:12]}: {entry['path']}")
        if item.size != entry["bytes"]:
            raise SystemExit(f"Tamaño distinto en {repo}: {entry['path']} ({item.size} frente a {entry['bytes']})")
        if item.lfs and item.lfs.sha256 != entry["sha256"]:
            raise SystemExit(f"sha256 distinto en {repo}: {entry['path']}")
    remote_manifest = Path(hf_hub_download(repo, "MANIFEST.json", repo_type="dataset", revision=commit))
    if sha256(remote_manifest) != manifest_sha256:
        raise SystemExit(f"MANIFEST.json de {repo} no coincide con el del paquete local.")
    print(f"Repositorio de datos verificado: {len(manifest['files'])} archivos y MANIFEST.json idénticos.")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description="Arma la carpeta del Space de Hugging Face del servicio /retrieve.")
    parser.add_argument("--data", type=Path, required=True, help="Carpeta storage del paquete de datos (con MANIFEST.json).")
    parser.add_argument("--data-repo", required=True, help="Repositorio de dataset privado con el paquete, p. ej. usuario/rag-istae-data-lb1.")
    parser.add_argument("--data-revision", help="Commit del repositorio de datos; por defecto, el actual de la rama principal.")
    parser.add_argument("--out", type=Path, required=True, help="Carpeta de salida (repositorio del Space).")
    parser.add_argument("--force", action="store_true", help="Reemplaza --out si ya existe.")
    args = parser.parse_args()

    data = args.data.resolve()
    out = args.out.resolve()
    if ROOT in out.parents or out == ROOT:
        raise SystemExit("--out no puede estar dentro del repositorio rag-istae-query.")
    if out.exists() and not args.force:
        raise SystemExit(f"{out} ya existe; use --force para reemplazarlo.")

    manifest = verify_bundle(data)
    manifest_sha256 = sha256(data / "MANIFEST.json")
    commit = verify_remote(args.data_repo, args.data_revision, manifest, manifest_sha256)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    dockerfile = (DEPLOY / "Dockerfile").read_text(encoding="utf-8")
    dockerfile = (dockerfile.replace("{data_repo}", args.data_repo).replace("{data_revision}", commit)
                  .replace("{manifest_sha256}", manifest_sha256))
    (out / "Dockerfile").write_text(dockerfile, encoding="utf-8", newline="\n")
    for name in ("requirements-space.txt", "fetch_data.py"):
        shutil.copy2(DEPLOY / name, out / name)
    readme = (DEPLOY / "README.space.md").read_text(encoding="utf-8")
    readme = (readme.replace("{bundle_version}", manifest["version"])
              .replace("{units_sha256}", manifest["index_version"]["units_sha256"])
              .replace("{data_repo}", args.data_repo).replace("{data_revision}", commit))
    (out / "README.md").write_text(readme, encoding="utf-8")
    shutil.copytree(ROOT / "src" / "query", out / "src" / "query",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    total = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    print(f"Space listo en {out} ({total / 1e6:.2f} MB, sin datos).")
    print(f"Datos: {args.data_repo} @ {commit} · paquete {manifest['version']} · "
          f"units_sha256 = {manifest['index_version']['units_sha256']}")


if __name__ == "__main__":
    sys.exit(main())

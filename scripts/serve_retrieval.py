"""Arranca el servicio de recuperación (``POST /retrieve``) con uvicorn.

Uso (desde la raíz de rag-istae-query, con el .venv activo):

    $env:RAG_API_KEY = "<clave>"            # o $env:RAG_AUTH_DISABLED = "true" solo en local
    python .\\scripts\\serve_retrieval.py [--host 127.0.0.1] [--port 8100]

La configuración se lee de variables de entorno (``query.service.settings``):
por defecto ``config/local.json``. Un solo worker: Qdrant local admite un solo
proceso. No ejecute scripts de consulta sobre el mismo Qdrant mientras el
servicio está activo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Servicio de recuperación RAG ISTAE (B/C).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "query.service.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

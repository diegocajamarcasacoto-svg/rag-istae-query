"""Configuración del servicio por variables de entorno (sin rutas fijas en el código).

- ``RAG_CONFIG``: JSON con el formato de ``config/local.json`` (por defecto, el
  ``config/local.json`` del repositorio si existe).
- Cada ruta o valor puede sobrescribirse con una variable propia, para
  contenedores u otro host: ``RAG_QDRANT_PATH``, ``RAG_E5_MODEL_PATH``,
  ``RAG_RERANKER_MODEL_PATH``, ``RAG_UNITS_PATH``, ``RAG_CATALOG_PATH``,
  ``RAG_COLLECTION``.
- ``RAG_API_KEY``: clave que deben enviar los clientes en ``X-API-Key``.
  Obligatoria salvo ``RAG_AUTH_DISABLED=true`` (solo pruebas locales).
- ``RAG_LOG_DIR`` (por defecto ``logs/service``) y ``RAG_CACHE_SIZE`` (128).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from query.config import REQUIRED

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "local.json"
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "service"

ENV_OVERRIDES = {
    "RAG_QDRANT_PATH": "qdrant_path",
    "RAG_E5_MODEL_PATH": "e5_model_path",
    "RAG_RERANKER_MODEL_PATH": "reranker_model_path",
    "RAG_UNITS_PATH": "units_path",
    "RAG_CATALOG_PATH": "catalog_path",
    "RAG_COLLECTION": "collection_name",
}
PATH_KEYS = ("e5_model_path", "reranker_model_path", "qdrant_path", "units_path", "catalog_path")
TRUE_VALUES = {"1", "true", "yes", "si", "sí"}


@dataclass(frozen=True)
class ServiceSettings:
    retrieval_config: dict[str, Any]
    api_key: str | None
    auth_disabled: bool
    log_dir: Path | None
    cache_size: int

    def describe(self) -> dict[str, Any]:
        """Resumen sin secretos (para /health y los registros)."""
        return {
            "auth": "disabled" if self.auth_disabled else "api_key",
            "log_dir": str(self.log_dir) if self.log_dir else None,
            "cache_size": self.cache_size,
        }


def retrieval_config_from_env(env: Mapping[str, str]) -> dict[str, Any]:
    config_path = Path(env["RAG_CONFIG"]) if env.get("RAG_CONFIG") else DEFAULT_CONFIG
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    elif env.get("RAG_CONFIG"):
        raise FileNotFoundError(f"RAG_CONFIG no existe: {config_path}")
    for variable, key in ENV_OVERRIDES.items():
        if env.get(variable):
            config[key] = env[variable]
    missing = [key for key in (*REQUIRED, "units_path", "catalog_path") if not config.get(key)]
    if missing:
        raise KeyError(
            "Faltan claves de configuración del servicio: " + ", ".join(missing)
            + ". Indique RAG_CONFIG o las variables RAG_* correspondientes."
        )
    for key in PATH_KEYS:
        config[key] = Path(config[key]).expanduser().resolve()
    return config


def load_settings(env: Mapping[str, str] | None = None) -> ServiceSettings:
    env = os.environ if env is None else env
    auth_disabled = env.get("RAG_AUTH_DISABLED", "").strip().lower() in TRUE_VALUES
    api_key = env.get("RAG_API_KEY") or None
    if not api_key and not auth_disabled:
        raise ValueError("Defina RAG_API_KEY (o RAG_AUTH_DISABLED=true solo para pruebas locales).")
    log_value = env.get("RAG_LOG_DIR")
    cache_size = int(env.get("RAG_CACHE_SIZE", "128"))
    if cache_size < 0:
        raise ValueError("RAG_CACHE_SIZE no puede ser negativo.")
    return ServiceSettings(
        retrieval_config=retrieval_config_from_env(env),
        api_key=api_key,
        auth_disabled=auth_disabled,
        log_dir=None if log_value == "-" else Path(log_value) if log_value else DEFAULT_LOG_DIR,
        cache_size=cache_size,
    )

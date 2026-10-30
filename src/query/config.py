from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED = ("e5_model_path", "reranker_model_path", "qdrant_path", "collection_name")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED if not config.get(key)]
    if missing:
        raise KeyError(f"Faltan claves de configuracion: {', '.join(missing)}")
    for key in ("e5_model_path", "reranker_model_path", "qdrant_path"):
        config[key] = Path(config[key]).expanduser().resolve()
    return config

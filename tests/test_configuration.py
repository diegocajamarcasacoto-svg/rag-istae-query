import json
from pathlib import Path

from query.config import load_config


def test_configuration_resolves_absolute_paths(tmp_path: Path) -> None:
    path = tmp_path / "local.json"
    path.write_text(json.dumps({"e5_model_path": "e5", "reranker_model_path": "bge", "qdrant_path": "qdrant", "collection_name": "test"}), encoding="utf-8")
    config = load_config(path)
    assert config["collection_name"] == "test"
    assert config["e5_model_path"].is_absolute()

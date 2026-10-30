from __future__ import annotations

from pathlib import Path

from sentence_transformers import CrossEncoder


def load_reranker(path: Path) -> CrossEncoder:
    if not path.is_dir() or not any(path.iterdir()):
        raise FileNotFoundError(f"No existe el modelo reranker local: {path}")
    return CrossEncoder(str(path), device="cpu", max_length=512)


def rerank(model: CrossEncoder, query: str, candidates, batch_size: int):
    valid = [point for point in candidates if str((point.payload or {}).get("text") or "").strip()]
    scores = model.predict([[query, (point.payload or {})["text"]] for point in valid], batch_size=batch_size, show_progress_bar=False)
    return sorted(zip(valid, scores), key=lambda item: float(item[1]), reverse=True)

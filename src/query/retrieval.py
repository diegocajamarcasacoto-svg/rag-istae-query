from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


def load_e5(config: dict[str, Any]) -> SentenceTransformer:
    path = config["e5_model_path"]
    if not path.is_dir() or not any(path.iterdir()):
        raise FileNotFoundError(f"No existe el modelo E5 local: {path}")
    model = SentenceTransformer(str(path), device="cpu")
    model.max_seq_length = int(config.get("max_seq_length", 512))
    return model


def encode_query(model: SentenceTransformer, text: str, config: dict[str, Any]) -> list[float]:
    return model.encode(
        "query: " + text.strip(),
        convert_to_numpy=True,
        normalize_embeddings=bool(config.get("normalize_embeddings", True)),
    ).tolist()


def open_client(config: dict[str, Any]) -> QdrantClient:
    path = config["qdrant_path"]
    if not path.exists():
        raise FileNotFoundError(f"No existe la ruta Qdrant: {path}")
    client = QdrantClient(path=str(path))
    if not client.collection_exists(config["collection_name"]):
        client.close()
        raise RuntimeError(f"No existe la coleccion Qdrant: {config['collection_name']}")
    return client


def vector_search(client: QdrantClient, config: dict[str, Any], vector: list[float], limit: int):
    return client.query_points(collection_name=config["collection_name"], query=vector, limit=limit, with_payload=True).points


def bm25_search(client: QdrantClient, config: dict[str, Any], text: str, limit: int):
    return client.query_points(collection_name=config["collection_name"], query=models.TextQuery(text=text), using="text", limit=limit, with_payload=True).points


def hybrid_search(client: QdrantClient, config: dict[str, Any], vector: list[float], text: str, prefetch_k: int, candidate_k: int, rrf_k: int):
    return client.query_points(
        collection_name=config["collection_name"],
        prefetch=[models.Prefetch(query=vector, limit=prefetch_k), models.Prefetch(query=models.TextQuery(text=text), using="text", limit=prefetch_k)],
        query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
        limit=candidate_k,
        with_payload=True,
    ).points

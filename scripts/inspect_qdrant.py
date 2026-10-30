import argparse
import sys
from pathlib import Path

from qdrant_client import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from query.config import load_config
from query.retrieval import open_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspecciona Qdrant sin modificarlo.")
    parser.add_argument("--config", type=Path, default=ROOT / "config/local.json")
    parser.add_argument("--chunk-id")
    parser.add_argument("--neighbors", type=int, default=1)
    args = parser.parse_args()
    if args.neighbors < 0: parser.error("--neighbors no puede ser negativo.")
    config = load_config(args.config); client = open_client(config)
    try:
        collection = config["collection_name"]
        info = client.get_collection(collection)
        print(f"Coleccion: {collection}\nPuntos: {info.points_count}\nVectores: {info.config.params.vectors}")
        if not args.chunk_id: return
        points, _ = client.scroll(collection_name=collection, scroll_filter=models.Filter(must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value=args.chunk_id))]), limit=1, with_payload=True, with_vectors=False)
        if not points: raise SystemExit(f"No se encontro el fragmento: {args.chunk_id}")
        payload = points[0].payload or {}; document_id = payload.get("document_id"); index = payload.get("chunk_index")
        if not isinstance(index, int) or isinstance(index, bool) or not document_id: raise SystemExit("El fragmento no tiene document_id/chunk_index validos.")
        related, _ = client.scroll(collection_name=collection, scroll_filter=models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)), models.FieldCondition(key="chunk_index", range=models.Range(gte=max(1, index-args.neighbors), lte=index+args.neighbors))]), limit=(args.neighbors * 2) + 1, with_payload=True, with_vectors=False)
        for point in sorted(related, key=lambda item: (item.payload or {}).get("chunk_index", 0)):
            item = point.payload or {}; print(f"\n{item.get('chunk_id')} | indice={item.get('chunk_index')}\n{item.get('text', '')}")
    finally: client.close()


if __name__ == "__main__": main()

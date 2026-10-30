import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from query.config import load_config
from query.retrieval import bm25_search, open_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Busqueda BM25 nativa de Qdrant.")
    parser.add_argument("query")
    parser.add_argument("--config", type=Path, default=ROOT / "config/local.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1: parser.error("--top-k debe ser positivo.")
    config = load_config(args.config)
    client = open_client(config)
    try:
        for rank, point in enumerate(bm25_search(client, config, args.query, args.top_k), 1):
            payload = point.payload or {}
            print(f"#{rank} | score={point.score:.6f} | chunk_id={payload.get('chunk_id')}\n{payload.get('text', '')[:1000]}\n")
    finally:
        client.close()


if __name__ == "__main__": main()

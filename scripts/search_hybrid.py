import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from query.config import load_config
from query.retrieval import encode_query, hybrid_search, load_e5, open_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Busqueda E5 + BM25 Qdrant + RRF.")
    parser.add_argument("query")
    parser.add_argument("--config", type=Path, default=ROOT / "config/local.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prefetch-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--offline", action="store_true", help="Compatibilidad: Query siempre exige modelos locales.")
    args = parser.parse_args()
    if min(args.top_k, args.prefetch_k, args.rrf_k) < 1: parser.error("Los limites deben ser positivos.")
    config = load_config(args.config); model = load_e5(config); client = open_client(config)
    try:
        results = hybrid_search(client, config, encode_query(model, args.query, config), args.query, args.prefetch_k, args.top_k, args.rrf_k)
        for rank, point in enumerate(results, 1):
            payload = point.payload or {}
            print(f"#{rank} | RRF={point.score:.6f} | chunk_id={payload.get('chunk_id')}\n{payload.get('text', '')[:1000]}\n")
    finally: client.close()


if __name__ == "__main__": main()

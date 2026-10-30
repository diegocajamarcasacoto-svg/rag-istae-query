import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from query.config import load_config
from query.reranking import load_reranker, rerank
from query.retrieval import encode_query, hybrid_search, load_e5, open_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Busqueda E5 + BM25 + RRF + BGE reranker.")
    parser.add_argument("query")
    parser.add_argument("--config", type=Path, default=ROOT / "config/local.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--prefetch-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reranker-model", type=Path, help="Ruta local BGE; por defecto usa la configuracion.")
    args = parser.parse_args()
    if min(args.top_k, args.candidate_k, args.prefetch_k, args.rrf_k, args.batch_size) < 1 or args.top_k > args.candidate_k: parser.error("Parametros de recuperacion invalidos.")
    config = load_config(args.config); e5 = load_e5(config); reranker = load_reranker(args.reranker_model or config["reranker_model_path"]); client = open_client(config)
    try:
        candidates = hybrid_search(client, config, encode_query(e5, args.query, config), args.query, args.prefetch_k, args.candidate_k, args.rrf_k)
        for rank, (point, score) in enumerate(rerank(reranker, args.query, candidates, args.batch_size)[:args.top_k], 1):
            payload = point.payload or {}
            print(f"#{rank} | reranker={float(score):.6f} | RRF={point.score:.6f} | chunk_id={payload.get('chunk_id')}\n{payload.get('text', '')[:1000]}\n")
    finally: client.close()


if __name__ == "__main__": main()

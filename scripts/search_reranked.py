import argparse
import sys
import time
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

    process_start = time.perf_counter()
    t0 = time.perf_counter()
    config = load_config(args.config)
    config_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    e5 = load_e5(config)
    e5_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    reranker = load_reranker(args.reranker_model or config["reranker_model_path"])
    bge_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    client = open_client(config)
    qdrant_open_seconds = time.perf_counter() - t0
    initialization_total_seconds = time.perf_counter() - process_start

    try:
        query_start = time.perf_counter()
        t0 = time.perf_counter()
        query_vector = encode_query(e5, args.query, config)
        query_embedding_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        candidates = hybrid_search(client, config, query_vector, args.query, args.prefetch_k, args.candidate_k, args.rrf_k)
        hybrid_rrf_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        ranked_results = rerank(reranker, args.query, candidates, args.batch_size)
        reranking_seconds = time.perf_counter() - t0
        results = ranked_results[:args.top_k]
        query_total_seconds = time.perf_counter() - query_start
        process_total_seconds = time.perf_counter() - process_start
    finally: client.close()

    for rank, (point, score) in enumerate(results, 1):
        payload = point.payload or {}
        print(f"#{rank} | reranker={float(score):.6f} | RRF={point.score:.6f} | chunk_id={payload.get('chunk_id')}\n{payload.get('text', '')[:1000]}\n")

    print("--- MÉTRICAS DE RENDIMIENTO ---")
    print("Inicialización")
    print(f"Configuración:        {config_load_seconds:.3f} s")
    print(f"Carga E5:             {e5_load_seconds:.3f} s")
    print(f"Carga BGE:            {bge_load_seconds:.3f} s")
    print(f"Apertura Qdrant:      {qdrant_open_seconds:.3f} s")
    print(f"Total inicialización: {initialization_total_seconds:.3f} s")
    print("\nConsulta")
    print(f"Embedding:            {query_embedding_seconds:.3f} s")
    print(f"Híbrida + RRF:        {hybrid_rrf_seconds:.3f} s")
    print(f"Reranking:            {reranking_seconds:.3f} s")
    print(f"Total consulta:       {query_total_seconds:.3f} s")
    print(f"\nProceso total:       {process_total_seconds:.3f} s")


if __name__ == "__main__": main()

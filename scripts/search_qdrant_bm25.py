import argparse
import sys
import time
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

    process_start = time.perf_counter()
    t0 = time.perf_counter()
    config = load_config(args.config)
    config_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    client = open_client(config)
    qdrant_open_seconds = time.perf_counter() - t0
    initialization_total_seconds = time.perf_counter() - process_start

    try:
        query_start = time.perf_counter()
        t0 = time.perf_counter()
        results = bm25_search(client, config, args.query, args.top_k)
        bm25_search_seconds = time.perf_counter() - t0
        query_total_seconds = time.perf_counter() - query_start
        process_total_seconds = time.perf_counter() - process_start
    finally:
        client.close()

    for rank, point in enumerate(results, 1):
        payload = point.payload or {}
        print(f"#{rank} | score={point.score:.6f} | chunk_id={payload.get('chunk_id')}\n{payload.get('text', '')[:1000]}\n")

    print("--- MÉTRICAS DE RENDIMIENTO ---")
    print("Inicialización")
    print(f"Configuración:        {config_load_seconds:.3f} s")
    print(f"Apertura Qdrant:      {qdrant_open_seconds:.3f} s")
    print(f"Total inicialización: {initialization_total_seconds:.3f} s")
    print("\nConsulta")
    print(f"Búsqueda BM25:        {bm25_search_seconds:.3f} s")
    print(f"Total consulta:       {query_total_seconds:.3f} s")
    print(f"\nProceso total:       {process_total_seconds:.3f} s")


if __name__ == "__main__": main()

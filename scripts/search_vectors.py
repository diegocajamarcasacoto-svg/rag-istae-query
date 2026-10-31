import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from query.config import load_config
from query.retrieval import encode_query, load_e5, open_client, vector_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Busqueda semantica E5 en Qdrant.")
    parser.add_argument("query")
    parser.add_argument("--config", type=Path, default=ROOT / "config/local.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--offline", action="store_true", help="Compatibilidad: Query siempre exige modelos locales.")
    args = parser.parse_args()
    if args.top_k < 1: parser.error("--top-k debe ser positivo.")

    process_start = time.perf_counter()
    t0 = time.perf_counter()
    config = load_config(args.config)
    config_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = load_e5(config)
    e5_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    client = open_client(config)
    qdrant_open_seconds = time.perf_counter() - t0
    initialization_total_seconds = time.perf_counter() - process_start

    try:
        query_start = time.perf_counter()
        t0 = time.perf_counter()
        query_vector = encode_query(model, args.query, config)
        query_embedding_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        results = vector_search(client, config, query_vector, args.top_k)
        vector_search_seconds = time.perf_counter() - t0
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
    print(f"Carga E5:             {e5_load_seconds:.3f} s")
    print(f"Apertura Qdrant:      {qdrant_open_seconds:.3f} s")
    print(f"Total inicialización: {initialization_total_seconds:.3f} s")
    print("\nConsulta")
    print(f"Embedding:            {query_embedding_seconds:.3f} s")
    print(f"Búsqueda vectorial:   {vector_search_seconds:.3f} s")
    print(f"Total consulta:       {query_total_seconds:.3f} s")
    print(f"\nProceso total:       {process_total_seconds:.3f} s")


if __name__ == "__main__": main()

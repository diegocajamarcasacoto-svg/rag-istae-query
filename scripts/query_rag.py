from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.config import load_config
from query.context import build_context
from query.generation import generate_answer, load_generation_config
from query.reranking import load_reranker, rerank
from query.retrieval import encode_query, hybrid_search, load_e5, open_client


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el pipeline RAG completo: recuperación híbrida, reranking "
            "opcional, construcción de contexto y generación con Gemini."
        )
    )
    parser.add_argument("query", help="Pregunta que se responderá.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "local.json",
        help="Configuración local de E5 y Qdrant.",
    )
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=ROOT / "config" / "generation.json",
        help="Configuración independiente del módulo de generación.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--prefetch-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reranker-model", type=Path)
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activa o desactiva el reranker BGE (predeterminado: activado).",
    )
    parser.add_argument("--seed-k", type=int, default=3)
    parser.add_argument("--neighbors-before", type=int, default=1)
    parser.add_argument("--neighbors-after", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Ruta opcional para exportar respuesta, contexto, procedencia y métricas.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "top-k": args.top_k,
        "candidate-k": args.candidate_k,
        "prefetch-k": args.prefetch_k,
        "rrf-k": args.rrf_k,
        "batch-size": args.batch_size,
        "seed-k": args.seed_k,
    }
    if any(value < 1 for value in positive.values()):
        parser.error("top-k, candidate-k, prefetch-k, rrf-k, batch-size y seed-k deben ser positivos.")
    if args.top_k > args.candidate_k:
        parser.error("--top-k no puede superar --candidate-k.")
    if args.neighbors_before < 0 or args.neighbors_after < 0:
        parser.error("--neighbors-before y --neighbors-after no pueden ser negativos.")
    if args.max_context_tokens is not None and args.max_context_tokens < 1:
        parser.error("--max-context-tokens debe ser positivo.")


def _chunk_id(point: Any) -> str | None:
    payload = point.payload if isinstance(point.payload, dict) else {}
    value = payload.get("chunk_id")
    return value if isinstance(value, str) else None


def _print_result(
    *,
    answer: Any,
    candidates: list[Any],
    ranked_results: list[tuple[Any, float | None]],
    rerank_enabled: bool,
    timings: dict[str, float | None],
) -> None:
    context_result = answer.context_result
    print("\n=== RESPUESTA ===")
    print(answer.answer)

    print("\n=== FUENTES DEL CONTEXTO ===")
    if not context_result.chunks:
        print("No se seleccionaron chunks de contexto.")
    for chunk in context_result.chunks:
        pages = f", páginas {chunk.page_numbers}" if chunk.page_numbers else ""
        print(
            f"- {chunk.source_file or chunk.document_id or 'origen desconocido'}"
            f"{pages}; chunk={chunk.chunk_id or 'sin ID'}; rol={chunk.role}"
        )

    print("\n=== RESUMEN DEL PIPELINE ===")
    print(f"Candidatos híbridos: {len(candidates)}")
    print(f"Chunks enviados al contexto: {len(context_result.chunks)}")
    print(f"Reranker: {'activado' if rerank_enabled else 'desactivado (orden RRF)'}")
    print(f"Proveedor/modelo: {answer.provider}/{answer.model}")
    print(
        "Tokens reportados por el proveedor: "
        f"entrada={answer.provider_metrics.input_tokens}, "
        f"salida={answer.provider_metrics.output_tokens}"
    )
    print(f"Latencia de generación medida por la aplicación: {answer.application_latency_ms:.3f} ms")
    print("\nTiempos de ejecución:")
    for stage, seconds in timings.items():
        print(f"- {stage}: {seconds:.3f} s" if seconds is not None else f"- {stage}: no ejecutado")

    if ranked_results:
        ranking_source = "reranker" if rerank_enabled else "RRF"
        print(f"\nTop de selección ({ranking_source}):")
        for rank, (point, score) in enumerate(ranked_results, 1):
            ranking_score = score if rerank_enabled else getattr(point, "score", None)
            print(f"- #{rank} chunk={_chunk_id(point)} score={ranking_score}")


def _export_json(
    path: Path,
    *,
    query: str,
    answer: Any,
    candidates: list[Any],
    ranked_results: list[tuple[Any, float | None]],
    rerank_enabled: bool,
    timings: dict[str, float | None],
) -> None:
    payload = {
        "query": query,
        "generation_result": answer.to_dict(),
        "retrieval_results": [
            {"chunk_id": _chunk_id(point), "rrf_score": getattr(point, "score", None)}
            for point in candidates
        ],
        "ranking_results": [
            {
                "chunk_id": _chunk_id(point),
                "ranking_source": "reranker" if rerank_enabled else "rrf",
                "ranking_score": (
                    float(score) if rerank_enabled and score is not None
                    else getattr(point, "score", None)
                ),
            }
            for point, score in ranked_results
        ],
        "timings_seconds": timings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    process_start = time.perf_counter()
    stage_start = time.perf_counter()
    config = load_config(args.config)
    generation_config = load_generation_config(args.generation_config)
    configuration_seconds = time.perf_counter() - stage_start
    if not os.environ.get(generation_config.api_key_env):
        parser.error(
            f"Define la variable de entorno {generation_config.api_key_env} "
            "antes de ejecutar una consulta."
        )

    stage_start = time.perf_counter()
    e5 = load_e5(config)
    e5_load_seconds = time.perf_counter() - stage_start

    reranker = None
    reranker_load_seconds = None
    if args.rerank:
        stage_start = time.perf_counter()
        reranker = load_reranker(args.reranker_model or config["reranker_model_path"])
        reranker_load_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    client = open_client(config)
    qdrant_open_seconds = time.perf_counter() - stage_start

    timings: dict[str, float | None] = {
        "configuration_load": configuration_seconds,
        "e5_load": e5_load_seconds,
        "reranker_load": reranker_load_seconds,
        "qdrant_open": qdrant_open_seconds,
    }
    try:
        stage_start = time.perf_counter()
        query_vector = encode_query(e5, args.query, config)
        timings["query_embedding"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        candidates = hybrid_search(
            client,
            config,
            query_vector,
            args.query,
            args.prefetch_k,
            args.candidate_k,
            args.rrf_k,
        )
        timings["hybrid_retrieval_and_rrf"] = time.perf_counter() - stage_start
        if not candidates:
            raise RuntimeError("La recuperación híbrida no devolvió candidatos.")

        if args.rerank:
            stage_start = time.perf_counter()
            reranked = rerank(reranker, args.query, candidates, args.batch_size)
            timings["reranking"] = time.perf_counter() - stage_start
            ranked_results = reranked[:args.top_k]
        else:
            timings["reranking"] = None
            ranked_results = [(point, None) for point in candidates[:args.top_k]]
        if not ranked_results:
            raise RuntimeError("No hay candidatos con texto para construir el contexto.")

        builder_config = {
            **config,
            "context_builder": {
                "seed_count": args.seed_k,
                "neighbors_before": args.neighbors_before,
                "neighbors_after": args.neighbors_after,
                "require_same_document": True,
                "require_consecutive": True,
                "max_tokens": args.max_context_tokens,
            },
        }
        stage_start = time.perf_counter()
        context_result = build_context(args.query, ranked_results, client, builder_config)
        timings["context_builder"] = time.perf_counter() - stage_start
    finally:
        client.close()

    stage_start = time.perf_counter()
    answer = generate_answer(context_result, generation_config)
    timings["generation"] = time.perf_counter() - stage_start
    timings["total"] = time.perf_counter() - process_start

    _print_result(
        answer=answer,
        candidates=candidates,
        ranked_results=ranked_results,
        rerank_enabled=args.rerank,
        timings=timings,
    )
    if args.json_output:
        _export_json(
            args.json_output,
            query=args.query,
            answer=answer,
            candidates=candidates,
            ranked_results=ranked_results,
            rerank_enabled=args.rerank,
            timings=timings,
        )
        print(f"\nResultado JSON exportado: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

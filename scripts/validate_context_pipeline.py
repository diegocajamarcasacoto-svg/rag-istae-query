from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.config import load_config
from query.context import build_context
from query.reranking import load_reranker, rerank
from query.retrieval import encode_query, hybrid_search, load_e5, open_client


def _chunk_id(point: Any) -> str | None:
    payload = point.payload or {}
    value = payload.get("chunk_id")
    return value if isinstance(value, str) else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valida recuperación híbrida, reranking y construcción del contexto."
    )
    parser.add_argument("query")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "local.json")
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
    parser.add_argument("--json-output", type=Path)
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
        parser.error("La cantidad de vecinos no puede ser negativa.")
    if args.max_context_tokens is not None and args.max_context_tokens < 1:
        parser.error("--max-context-tokens debe ser positivo.")


def _configuration_summary(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "collection": config["collection_name"],
        "embedding_model": Path(config["e5_model_path"]).name,
        "reranker_enabled": args.rerank,
        "reranker_model": (
            Path(args.reranker_model or config["reranker_model_path"]).name
            if args.rerank
            else None
        ),
        "retrieval": {
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "prefetch_k": args.prefetch_k,
            "rrf_k": args.rrf_k,
            "batch_size": args.batch_size,
        },
        "context_builder": {
            "seed_count": args.seed_k,
            "neighbors_before": args.neighbors_before,
            "neighbors_after": args.neighbors_after,
            "require_same_document": True,
            "require_consecutive": True,
            "max_tokens": args.max_context_tokens,
        },
    }


def export_validation_json(path: Path, payload: dict[str, Any]) -> None:
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
    config_load_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    e5 = load_e5(config)
    e5_load_seconds = time.perf_counter() - stage_start

    reranker = None
    reranker_load_seconds = None
    if args.rerank:
        reranker_path = args.reranker_model or config["reranker_model_path"]
        stage_start = time.perf_counter()
        reranker = load_reranker(reranker_path)
        reranker_load_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    client = open_client(config)
    qdrant_open_seconds = time.perf_counter() - stage_start
    initialization_seconds = time.perf_counter() - process_start

    try:
        query_start = time.perf_counter()
        stage_start = time.perf_counter()
        query_vector = encode_query(e5, args.query, config)
        embedding_seconds = time.perf_counter() - stage_start

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
        hybrid_seconds = time.perf_counter() - stage_start

        if args.rerank:
            stage_start = time.perf_counter()
            reranker_results = rerank(reranker, args.query, candidates, args.batch_size)
            reranking_seconds = time.perf_counter() - stage_start
            context_ranking_results = reranker_results
        else:
            reranker_results = []
            reranking_seconds = None
            context_ranking_results = [(point, None) for point in candidates]

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
        selected_results = context_ranking_results[:args.top_k]
        context_result = build_context(args.query, selected_results, client, builder_config)
        query_processing_seconds = time.perf_counter() - query_start
        total_execution_seconds = time.perf_counter() - process_start
    finally:
        client.close()

    performance = {
        "configuration_load": config_load_seconds,
        "e5_load": e5_load_seconds,
        "reranker_load": reranker_load_seconds,
        "qdrant_open": qdrant_open_seconds,
        "initialization_total": initialization_seconds,
        "query_embedding": embedding_seconds,
        "hybrid_retrieval_and_rrf": hybrid_seconds,
        "reranking": reranking_seconds,
        **{
            f"context_{name}": seconds
            for name, seconds in context_result.metadata["timings_seconds"].items()
        },
        "query_processing_total": query_processing_seconds,
        "total_execution": total_execution_seconds,
    }
    summary = _configuration_summary(config, args)
    metrics = context_result.metadata["metrics"]

    print("A. CONFIGURACIÓN DE EJECUCIÓN")
    print(f"Colección Qdrant: {summary['collection']}")
    print(f"Modelo de embeddings: {summary['embedding_model']}")
    print(
        "Modelo reranker: "
        + (summary["reranker_model"] if args.rerank else "no cargado (--no-rerank)")
    )
    print(f"Recuperación: {summary['retrieval']}")
    print(f"Context Builder: {summary['context_builder']}")

    print("\nB. RECUPERACIÓN HÍBRIDA")
    print(f"Candidatos recuperados: {len(candidates)}")
    for rank, point in enumerate(candidates, 1):
        payload = point.payload or {}
        print(
            f"#{rank} chunk_id={payload.get('chunk_id')} "
            f"RRF={getattr(point, 'score', None)}"
        )

    print("\nC. RERANKING")
    if args.rerank:
        print(f"Resultados después del reranking: {len(reranker_results)}")
        for rank, (point, score) in enumerate(selected_results, 1):
            print(f"#{rank} chunk_id={_chunk_id(point)} reranker={float(score):.6f}")
    else:
        print("No ejecutado; se conserva el orden RRF de la recuperación híbrida.")
        for rank, (point, _) in enumerate(selected_results, 1):
            print(
                f"#{rank} chunk_id={_chunk_id(point)} "
                f"RRF={getattr(point, 'score', None)}"
            )
    seed_chunks = [chunk for chunk in context_result.chunks if chunk.role == "seed"]
    print(f"Semillas finales: {len(seed_chunks)}")
    for chunk in seed_chunks:
        score_label = (
            f"reranker={chunk.reranker_score:.6f}"
            if chunk.reranker_score is not None
            else "reranker=no ejecutado"
        )
        print(f"- {chunk.chunk_id} | {score_label}")

    print("\nD. CONSTRUCCIÓN DEL CONTEXTO")
    print(f"Vecinos recuperados: {metrics['neighbors_retrieved']}")
    print(f"Vecinos aceptados: {metrics['neighbors_accepted']}")
    print(f"Vecinos descartados: {metrics['neighbors_discarded']}")
    print(f"Complementarios incorporados: {metrics['complementary_chunks_included']}")
    print(f"Duplicados eliminados: {metrics['duplicates_removed']}")
    for duplicate in context_result.metadata["duplicate_chunks"]:
        print(
            f"- DUPLICADO {duplicate['chunk_id']} "
            f"motivo={duplicate['reason']} semillas={duplicate['included_by_seed_ids']}"
        )
    for chunk in context_result.chunks:
        if chunk.role != "seed":
            print(
                f"- {chunk.chunk_id} role={chunk.role} "
                f"seeds={chunk.included_by_seed_ids} reason={chunk.inclusion_reason}"
            )
    for discarded in context_result.metadata["discarded_chunks"]:
        print(
            f"- DESCARTADO {discarded['chunk_id']} "
            f"motivo={discarded['reason']} seed={discarded['seed_id']}"
        )

    print("\nE. CONTEXTO FINAL")
    print(f"Chunks finales: {metrics['final_chunk_count']}")
    token_label = "estimados" if metrics["token_count_is_estimated"] else "exactos"
    print(f"Tokens {token_label}: {metrics['token_count']}")
    print(f"IDs: {context_result.metadata['included_chunk_ids']}")
    print(f"Procedencia (documento -> páginas): {context_result.metadata['provenance']}")
    for chunk in context_result.chunks:
        print(
            f"\n[{chunk.chunk_id or 'sin chunk_id'} | {chunk.source_file or 'origen desconocido'}"
            f" | páginas={chunk.page_numbers} | rol={chunk.role}]"
        )
        print(chunk.text)
    print("\n--- TEXTO CONSOLIDADO ---")
    print(context_result.context)

    print("\nF. RENDIMIENTO (segundos)")
    for stage, seconds in performance.items():
        value = f"{seconds:.6f}" if seconds is not None else "no ejecutado"
        print(f"{stage:34} {value}")
    print("Nota: las etapas context_* son el desglose interno incluido en el tiempo total de Context Builder.")

    export_payload = {
        "query": args.query,
        "configuration": summary,
        "retrieval_results": [
            {
                "chunk_id": _chunk_id(point),
                "score": getattr(point, "score", None),
            }
            for point in candidates
        ],
        "reranker_results": [
            {"chunk_id": _chunk_id(point), "reranker_score": float(score)}
            for point, score in reranker_results
        ],
        "context_selection_results": [
            {
                "chunk_id": _chunk_id(point),
                "ranking_source": "reranker" if args.rerank else "rrf",
                "ranking_score": (
                    float(score) if score is not None else getattr(point, "score", None)
                ),
            }
            for point, score in context_ranking_results
        ],
        "seeds": [
            chunk.to_dict()
            for chunk in context_result.chunks
            if chunk.role == "seed"
        ],
        "included_chunks": [chunk.to_dict() for chunk in context_result.chunks],
        "discarded_chunks": context_result.metadata["discarded_chunks"],
        "context": context_result.context,
        "metadata": context_result.metadata,
        "performance_seconds": performance,
    }
    if args.json_output:
        export_validation_json(args.json_output, export_payload)
        print(f"\nJSON exportado: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

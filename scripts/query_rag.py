from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.config import load_config
from query.context import build_context
from query.generation import generate_answer, load_generation_config
from query.reranking import load_reranker, rerank
from query.retrieval import encode_query, hybrid_search, load_e5, open_client
from query.run_log import new_run_id, write_run
from query.units import build_unit_context, load_units, qdrant_chunk_fetcher

DEFAULT_UNITS_MAX_CONTEXT_TOKENS = 4000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el pipeline RAG completo: recuperación híbrida, reranking "
            "opcional, construcción de contexto y generación con el proveedor "
            "configurado en --generation-config (Gemini, OpenRouter, ...)."
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
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        help=(
            "Límite estimado de tokens del contexto. Sin límite en --expansion neighbors; "
            f"{DEFAULT_UNITS_MAX_CONTEXT_TOKENS} por defecto en --expansion units."
        ),
    )
    parser.add_argument(
        "--expansion",
        choices=("neighbors", "units"),
        default="neighbors",
        help=(
            "neighbors: Context Builder (semillas + vecinos heurísticos). "
            "units: cada chunk recuperado se sube a su artículo según units.jsonl (small-to-big)."
        ),
    )
    parser.add_argument(
        "--units-path",
        type=Path,
        help="units.jsonl generado por rag_istae_ingestion (por defecto, units_path de --config).",
    )
    parser.add_argument(
        "--max-unit-tokens",
        type=int,
        default=1500,
        help="En --expansion units, una unidad más larga se envía solo con sus chunks pertinentes.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Ruta opcional para exportar respuesta, contexto, procedencia y métricas.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs",
        help="Carpeta del registro automático de ejecuciones (predeterminado: logs/).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="No registrar esta ejecución en --log-dir.",
    )
    parser.add_argument(
        "--tag",
        help="Opcional: etiqueta libre del experimento (p. ej. nemotron-effort-none).",
    )
    parser.add_argument(
        "--question-id",
        help="Opcional: ID de la pregunta del Gold Standard (p. ej. P01).",
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
    if args.max_unit_tokens < 1:
        parser.error("--max-unit-tokens debe ser positivo.")


def _units_path(args: argparse.Namespace, config: dict[str, Any]) -> Path | None:
    value = args.units_path or config.get("units_path")
    return Path(value).expanduser().resolve() if value else None


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
    expansion = context_result.metadata.get("expansion", "neighbors")
    label = "Unidades (artículos)" if expansion == "units" else "Chunks"
    print(f"{label} enviados al contexto: {len(context_result.chunks)}")
    tokens = (context_result.metadata.get("metrics") or {}).get("token_count")
    if tokens is not None:
        print(f"Tokens estimados del contexto: {tokens}")
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


def _build_payload(
    *,
    query: str,
    answer: Any | None,
    candidates: list[Any] | None,
    ranked_results: list[tuple[Any, float | None]] | None,
    rerank_enabled: bool,
    timings: dict[str, float | None],
) -> dict[str, Any]:
    """Resultado serializable; las partes no calculadas (por error) quedan en None."""
    return {
        "query": query,
        "generation_result": answer.to_dict() if answer is not None else None,
        "retrieval_results": (
            [
                {"chunk_id": _chunk_id(point), "rrf_score": getattr(point, "score", None)}
                for point in candidates
            ]
            if candidates is not None
            else None
        ),
        "ranking_results": (
            [
                {
                    "chunk_id": _chunk_id(point),
                    "ranking_source": "reranker" if rerank_enabled else "rrf",
                    "ranking_score": (
                        float(score) if rerank_enabled and score is not None
                        else getattr(point, "score", None)
                    ),
                }
                for point, score in ranked_results
            ]
            if ranked_results is not None
            else None
        ),
        "timings_seconds": timings,
    }


def _export_json(path: Path, **payload_kwargs: Any) -> None:
    payload = _build_payload(**payload_kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _generation_config_snapshot(config: Any) -> dict[str, Any]:
    """Configuración de generación sin secretos: solo el nombre de la variable."""
    headers = {
        key: value
        for key, value in config.extra_headers.items()
        if key.lower() != "authorization"
    }
    return {
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "system_prompt_path": str(config.system_prompt_path),
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "timeout_seconds": config.timeout_seconds,
        "api_key_env": config.api_key_env,
        "extra_headers": headers,
        "extra_body": config.extra_body,
    }


def _retrieval_config_snapshot(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "collection_name": config.get("collection_name"),
        "qdrant_path": config.get("qdrant_path"),
        "e5_model_path": config.get("e5_model_path"),
        "reranker_model_path": (
            str(args.reranker_model or config.get("reranker_model_path"))
            if args.rerank
            else None
        ),
        "expansion": args.expansion,
        "units_path": (
            str(_units_path(args, config)) if args.expansion == "units" else None
        ),
    }


def _args_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _log_run(
    args: argparse.Namespace,
    *,
    run_id: str,
    started_at: datetime,
    status: str,
    error: dict[str, Any] | None,
    state: dict[str, Any],
    config: dict[str, Any],
    generation_config: Any,
) -> None:
    """Escribe el registro de la ejecución; un fallo aquí solo genera un aviso."""
    if args.no_log:
        return
    record = {
        "run_id": run_id,
        "timestamp": started_at.isoformat(timespec="seconds"),
        "status": status,
        "error": error,
        "tag": args.tag,
        "question_id": args.question_id,
        **_build_payload(
            query=args.query,
            answer=state["answer"],
            candidates=state["candidates"],
            ranked_results=state["ranked_results"],
            rerank_enabled=args.rerank,
            timings=state["timings"],
        ),
        "args": _args_snapshot(args),
        "generation_config": _generation_config_snapshot(generation_config),
        "retrieval_config": _retrieval_config_snapshot(config, args),
    }
    try:
        detail_path = write_run(args.log_dir, record)
    except Exception as exc:  # el registro nunca debe ocultar el resultado o el error
        print(f"\nAviso: no se pudo registrar la ejecución: {exc}", file=sys.stderr)
        return
    print(f"\nEjecución registrada: {detail_path}")


def _run_pipeline(
    args: argparse.Namespace,
    config: dict[str, Any],
    generation_config: Any,
    state: dict[str, Any],
    process_start: float,
) -> None:
    """Ejecuta el pipeline y deja resultados parciales en ``state``."""
    timings = state["timings"]

    state["stage"] = "model_load"
    stage_start = time.perf_counter()
    e5 = load_e5(config)
    timings["e5_load"] = time.perf_counter() - stage_start

    reranker = None
    timings["reranker_load"] = None
    if args.rerank:
        stage_start = time.perf_counter()
        reranker = load_reranker(args.reranker_model or config["reranker_model_path"])
        timings["reranker_load"] = time.perf_counter() - stage_start

    unit_index = None
    if args.expansion == "units":
        state["stage"] = "units_load"
        units_path = _units_path(args, config)
        if units_path is None:
            raise ValueError("--expansion units requiere --units-path o units_path en la configuración.")
        stage_start = time.perf_counter()
        unit_index = load_units(units_path)
        timings["units_load"] = time.perf_counter() - stage_start

    state["stage"] = "qdrant_open"
    stage_start = time.perf_counter()
    client = open_client(config)
    timings["qdrant_open"] = time.perf_counter() - stage_start

    try:
        state["stage"] = "retrieval"
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
        state["candidates"] = candidates
        if not candidates:
            raise RuntimeError("La recuperación híbrida no devolvió candidatos.")

        state["stage"] = "reranking"
        if args.rerank:
            stage_start = time.perf_counter()
            reranked = rerank(reranker, args.query, candidates, args.batch_size)
            timings["reranking"] = time.perf_counter() - stage_start
            ranked_results = reranked[:args.top_k]
        else:
            timings["reranking"] = None
            ranked_results = [(point, None) for point in candidates[:args.top_k]]
        state["ranked_results"] = ranked_results
        if not ranked_results:
            raise RuntimeError("No hay candidatos con texto para construir el contexto.")

        state["stage"] = "context_builder"
        stage_start = time.perf_counter()
        if unit_index is not None:
            # Small-to-big: cada chunk recuperado se sube a su artículo.
            context_result = build_unit_context(
                args.query,
                ranked_results,
                unit_index,
                fetch_chunk_texts=qdrant_chunk_fetcher(client, config["collection_name"]),
                max_tokens=args.max_context_tokens or DEFAULT_UNITS_MAX_CONTEXT_TOKENS,
                max_unit_tokens=args.max_unit_tokens,
            )
        else:
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
            context_result = build_context(args.query, ranked_results, client, builder_config)
        timings["context_builder"] = time.perf_counter() - stage_start
    finally:
        client.close()

    state["stage"] = "generation"
    stage_start = time.perf_counter()
    try:
        state["answer"] = generate_answer(context_result, generation_config)
    finally:
        timings["generation"] = time.perf_counter() - stage_start
    timings["total"] = time.perf_counter() - process_start


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    started_at = datetime.now().astimezone()
    run_id = new_run_id(started_at)
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

    state: dict[str, Any] = {
        "stage": "model_load",
        "candidates": None,
        "ranked_results": None,
        "answer": None,
        "timings": {"configuration_load": configuration_seconds},
    }
    log_kwargs = dict(
        run_id=run_id,
        started_at=started_at,
        state=state,
        config=config,
        generation_config=generation_config,
    )
    try:
        _run_pipeline(args, config, generation_config, state, process_start)
    except Exception as exc:
        state["timings"]["total"] = time.perf_counter() - process_start
        _log_run(
            args,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc), "stage": state["stage"]},
            **log_kwargs,
        )
        raise

    payload_kwargs = dict(
        answer=state["answer"],
        candidates=state["candidates"],
        ranked_results=state["ranked_results"],
        rerank_enabled=args.rerank,
        timings=state["timings"],
    )
    _print_result(**payload_kwargs)
    if args.json_output:
        _export_json(args.json_output, query=args.query, **payload_kwargs)
        print(f"\nResultado JSON exportado: {args.json_output}")
    _log_run(args, status="ok", error=None, **log_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

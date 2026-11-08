"""Pipeline de recuperación por artículos (B y C), compartido por scripts y servicio.

    embedding E5 → híbrida (vector + BM25, RRF) → rerank BGE (opcional)
    → artículos (units.jsonl) → capa jerárquica (opcional, C) → presentación por documentos

Recibe los recursos ya cargados (modelos, cliente Qdrant, índices), así
``scripts/query_rag.py`` (un proceso por consulta) y el servicio ``/retrieve``
(recursos residentes) ejecutan exactamente las mismas etapas. No genera respuesta.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from query.baseline import Baseline
from query.context.models import ContextResult
from query.hierarchy import expand_with_hierarchy
from query.hierarchy.catalog import Catalog
from query.presentation import render_documents
from query.reranking import rerank as rerank_candidates
from query.retrieval import encode_query, hybrid_search
from query.units import UnitIndex, build_unit_context, qdrant_chunk_fetcher


@dataclass
class PipelineResult:
    context_result: ContextResult
    candidates: list[Any]
    ranked_results: list[tuple[Any, float | None]]
    timings: dict[str, float | None] = field(default_factory=dict)


def run_units_pipeline(
    question: str,
    *,
    e5: Any,
    reranker: Any | None,
    client: Any,
    config: dict[str, Any],
    unit_index: UnitIndex,
    catalog: Catalog | None,
    params: Baseline,
    rerank: bool = True,
    hierarchy: bool = False,
    document_ids: Sequence[str] | None = None,
    state: dict[str, Any] | None = None,
) -> PipelineResult:
    """Ejecuta B (``hierarchy=False``) o C (``hierarchy=True``) con los parámetros ``params``.

    ``document_ids`` restringe la búsqueda base; la capa jerárquica puede agregar
    normas superiores fuera del filtro, que es su propósito. ``state`` (opcional)
    recibe los resultados parciales y la etapa en curso, para registrar fallas.
    """
    if rerank and reranker is None:
        raise ValueError("rerank=True requiere el modelo reranker cargado.")
    if hierarchy and catalog is None:
        raise ValueError("La capa jerárquica requiere el catálogo normativo.")
    state = state if state is not None else {}
    timings: dict[str, float | None] = state.setdefault("timings", {})
    filter_ids = list(document_ids) if document_ids else None

    state["stage"] = "retrieval"
    stage_start = time.perf_counter()
    query_vector = encode_query(e5, question, config)
    timings["query_embedding"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    candidates = hybrid_search(
        client, config, query_vector, question,
        params.prefetch_k, params.candidate_k, params.rrf_k,
        document_ids=filter_ids,
    )
    timings["hybrid_retrieval_and_rrf"] = time.perf_counter() - stage_start
    state["candidates"] = candidates
    if not candidates:
        raise RuntimeError("La recuperación híbrida no devolvió candidatos.")

    state["stage"] = "reranking"
    if rerank:
        stage_start = time.perf_counter()
        ranked_results = rerank_candidates(reranker, question, candidates, params.rerank_batch_size)[: params.top_k]
        timings["reranking"] = time.perf_counter() - stage_start
    else:
        timings["reranking"] = None
        ranked_results = [(point, None) for point in candidates[: params.top_k]]
    state["ranked_results"] = ranked_results
    if not ranked_results:
        raise RuntimeError("No hay candidatos con texto para construir el contexto.")

    # Small-to-big: cada chunk recuperado se sube a su artículo.
    state["stage"] = "context_builder"
    stage_start = time.perf_counter()
    fetch_chunk_texts = qdrant_chunk_fetcher(client, config["collection_name"])
    context_result = build_unit_context(
        question, ranked_results, unit_index,
        fetch_chunk_texts=fetch_chunk_texts,
        max_tokens=params.max_context_tokens,
        max_unit_tokens=params.max_unit_tokens,
    )
    timings["context_builder"] = time.perf_counter() - stage_start

    if hierarchy:
        # Arquitectura C: misma pregunta, búsqueda acotada a las normas superiores.
        def search_superior(target_ids: list[str]) -> list[tuple[Any, float | None]]:
            groups = [target_ids] if params.hierarchy_search == "joint" else [[d] for d in target_ids]
            found = []
            for group in groups:
                found += hybrid_search(
                    client, config, query_vector, question,
                    max(params.prefetch_k, params.hierarchy_candidate_k), params.hierarchy_candidate_k,
                    params.rrf_k, document_ids=group,
                )
            if rerank:
                return rerank_candidates(reranker, question, found, params.rerank_batch_size)
            if params.hierarchy_search == "per-document":
                found.sort(key=lambda point: point.score or 0.0, reverse=True)
            return [(point, None) for point in found]

        state["stage"] = "hierarchy"
        stage_start = time.perf_counter()
        context_result = expand_with_hierarchy(
            question, context_result, unit_index, catalog, search_superior,
            relations=params.hierarchy_relations,
            max_depth=params.hierarchy_depth,
            max_units=params.hierarchy_max_units,
            max_per_document=params.hierarchy_max_per_document,
            min_score=params.hierarchy_min_score,
            max_tokens=params.hierarchy_max_tokens,
            max_unit_tokens=params.max_unit_tokens,
            fetch_chunk_texts=fetch_chunk_texts,
        )
        # Etapa propia: búsqueda filtrada + rerank + selección de normas superiores.
        timings["hierarchy"] = time.perf_counter() - stage_start
        timings["hierarchy_search"] = (context_result.metadata.get("timings_seconds") or {}).get("hierarchy_search")

    # Mismo formato que rag-istae-api: [unit_id] agrupados por documento.
    state["stage"] = "presentation"
    stage_start = time.perf_counter()
    context_result = render_documents(context_result, catalog)
    timings["presentation"] = time.perf_counter() - stage_start
    state["context_result"] = context_result
    return PipelineResult(context_result, list(candidates), list(ranked_results), timings)

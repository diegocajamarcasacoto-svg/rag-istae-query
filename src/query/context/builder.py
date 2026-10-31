"""Orquestación para convertir resultados ordenados en contexto auditable."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

from query.context.context_merger import chunk_identity, deduplicate_and_sort
from query.context.models import ContextChunk, ContextResult
from query.context.neighbor_expander import expand_neighbors
from query.context.structure_detector import detect_structure
from query.context.structure_validator import (
    DEFAULT_COMPLEMENTARY_TERMS,
    validate_neighbor,
)
from query.context.token_budget import apply_token_budget


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def _page_numbers(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [page for page in value if isinstance(page, int) and not isinstance(page, bool)]


def _make_chunk(
    point: Any,
    *,
    role: str,
    seed_id: str | None,
    reason: str,
    score: float | None = None,
) -> ContextChunk:
    payload = point.payload if isinstance(point.payload, Mapping) else {}
    raw_index = payload.get("chunk_index")
    index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else None
    raw_doc = payload.get("document_id")
    raw_id = payload.get("chunk_id")
    structure = detect_structure(payload)
    return ContextChunk(
        chunk_id=raw_id if isinstance(raw_id, str) and raw_id else None,
        document_id=raw_doc if isinstance(raw_doc, str) and raw_doc else None,
        chunk_index=index,
        source_file=payload.get("source_file") if isinstance(payload.get("source_file"), str) else None,
        page_numbers=_page_numbers(payload.get("page_numbers")),
        headings=_string_list(payload.get("headings")),
        item_types=_string_list(payload.get("item_types")),
        text=payload.get("text") if isinstance(payload.get("text"), str) else "",
        reranker_score=score,
        role=role,
        structural_type=str(structure["type"]),
        structural_id=structure["id"],
        structure_status=str(structure["status"]),
        included_by_seed_ids=[seed_id] if seed_id else [],
        inclusion_reason=[reason],
    )


def _seed_reference(chunk: ContextChunk, rank: int) -> str:
    if chunk.chunk_id:
        return chunk.chunk_id
    if chunk.document_id and chunk.chunk_index is not None:
        return f"{chunk.document_id}:{chunk.chunk_index}"
    return f"seed-rank-{rank}"


def _point_identity(point: Any) -> tuple[str, ...]:
    payload = point.payload if isinstance(point.payload, Mapping) else {}
    chunk_id = payload.get("chunk_id")
    if isinstance(chunk_id, str) and chunk_id:
        return ("chunk_id", chunk_id)
    doc_id = payload.get("document_id")
    index = payload.get("chunk_index")
    if doc_id is not None and isinstance(index, int) and not isinstance(index, bool):
        return ("document_index", str(doc_id), str(index))
    point_id = getattr(point, "id", None)
    if point_id is not None:
        return ("point_id", str(point_id))
    return ("payload", str(dict(payload)))


def _finite_score(value: Any, position: int) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"El score de reranking en la posición {position} no es numérico.") from exc
    if not math.isfinite(score):
        raise ValueError(f"El score de reranking en la posición {position} no es finito.")
    return score


def build_context(
    query: str,
    ranked_results: list[tuple[Any, float | None]],
    client: Any,
    config: dict[str, Any],
) -> ContextResult:
    """Construye contexto acotado y trazable a partir de resultados ordenados.

    Esta función solo construye el contexto: no abre Qdrant, no carga E5/BGE,
    no calcula embeddings ni ejecuta búsquedas vectoriales, BM25 o RRF.
    Valida las entradas, toma hasta ``context_builder.seed_count`` elementos
    conservando el orden recibido, mantiene sus scores opcionales y registra
    los IDs de las semillas como anclas de procedencia.

    Para cada semilla con ``document_id`` y ``chunk_index``, solicita a
    ``expand_neighbors`` una ventana de índices filtrada y con payloads
    completos. Si detecta una lista, literal o numeral, sigue ventanas vecinas
    mediante una búsqueda en anchura limitada por
    ``max_list_continuation_chunks`` por semilla. Cada punto se valida contra
    el ancla desde la que se expandió; registra como descartados los candidatos
    que cruzan de documento, no son consecutivos, señalan un cambio explícito
    de artículo/capítulo/sección o no justifican contenido complementario.

    Deduplica y ordena semillas y vecinos aceptados antes de aplicar el
    presupuesto por chunks completos. Une los textos no vacíos con líneas en
    blanco. La metadata incluye métricas funcionales, tiempos por etapa,
    registros descartados/duplicados, documentos/archivos/páginas de origen,
    relaciones con semillas, IDs incluidos y estado estimado/exacto del conteo
    de tokens.

    ``config["context_builder"]`` admite ``seed_count`` (3),
    ``neighbors_before``/``neighbors_after`` (1), ``require_same_document`` y
    ``require_consecutive`` (ambos true), ``max_list_continuation_chunks``
    (20), ``complementary_query_terms``, ``max_tokens``, ``tokenizer`` y
    ``token_counter``. ``config["collection_name"]`` identifica la colección
    existente. No se escriben payloads ni metadata de colección.

    Args:
        query: Consulta original no vacía, usada también para evaluar notas y
            concordancias condicionales.
        ranked_results: Lista de pares ``(point, score)`` en el orden deseado.
            El score puede ser ``None`` si el proceso anterior no ejecutó un
            reranker; aquí no se reordena ni se vuelve a ejecutar un modelo.
        client: Cliente Qdrant ya abierto, utilizado solo para leer vecinos
            con filtros.
        config: Configuración con ``collection_name`` y opciones
            ``context_builder`` opcionales.

    Returns:
        ``ContextResult`` con texto consolidado, chunks incluidos y metadata
        de trazabilidad y rendimiento.

    Raises:
        TypeError: Si las formas de resultados/configuración o tipos de
            opciones son inválidos.
        ValueError: Si la consulta, colección, opciones numéricas o scores no
            son válidos.
        Excepciones del cliente Qdrant: Se propagan cuando falla la lectura de
            vecinos.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query debe ser un texto no vacío.")
    if not isinstance(ranked_results, list):
        raise TypeError("ranked_results debe ser una lista de pares (point, score).")
    if not isinstance(config, dict):
        raise TypeError("config debe ser un diccionario.")
    collection_name = config.get("collection_name")
    if not isinstance(collection_name, str) or not collection_name:
        raise ValueError("config debe incluir collection_name.")

    options = dict(config.get("context_builder", {}))
    seed_count = options.get("seed_count", 3)
    neighbors_before = options.get("neighbors_before", 1)
    neighbors_after = options.get("neighbors_after", 1)
    max_list_continuation_chunks = options.get("max_list_continuation_chunks", 20)
    for name, value in (
        ("seed_count", seed_count),
        ("neighbors_before", neighbors_before),
        ("neighbors_after", neighbors_after),
        ("max_list_continuation_chunks", max_list_continuation_chunks),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} debe ser un entero no negativo.")
    if seed_count < 1:
        raise ValueError("seed_count debe ser mayor que cero.")
    require_same_document = options.get("require_same_document", True)
    require_consecutive = options.get("require_consecutive", True)
    if not isinstance(require_same_document, bool) or not isinstance(require_consecutive, bool):
        raise TypeError("Las opciones de documento y consecutividad deben ser booleanas.")
    complementary_terms = options.get(
        "complementary_query_terms", DEFAULT_COMPLEMENTARY_TERMS
    )
    if not isinstance(complementary_terms, (list, tuple)) or not all(
        isinstance(term, str) for term in complementary_terms
    ):
        raise TypeError("complementary_query_terms debe ser una lista de textos.")

    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    selected_results = ranked_results[:seed_count]
    seeds: list[ContextChunk] = []
    seed_identities: set[tuple[str, ...]] = set()
    for rank, item in enumerate(selected_results, 1):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"ranked_results[{rank - 1}] debe ser un par (point, score).")
        point, raw_score = item
        score = _finite_score(raw_score, rank)
        seed = _make_chunk(
            point,
            role="seed",
            seed_id=None,
            reason=f"reranker_seed_rank_{rank}",
            score=score,
        )
        seed.included_by_seed_ids = [_seed_reference(seed, rank)]
        seeds.append(seed)
        seed_identities.add(_point_identity(point))
    timings["seed_selection"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    retrieved: dict[tuple[str, ...], Any] = {}
    candidate_pairs: list[tuple[ContextChunk, Any, str, ContextChunk]] = []
    warnings: list[str] = []
    for rank, seed in enumerate(seeds, 1):
        if not seed.document_id or seed.chunk_index is None:
            warnings.append(
                f"Semilla {_seed_reference(seed, rank)} sin document_id o chunk_index; no se expandieron vecinos."
            )
            continue
        seed_reference = _seed_reference(seed, rank)
        queue = deque([(seed, neighbors_before, neighbors_after)])
        visited_indexes = {seed.chunk_index}
        followed_list_chunks = 0
        while queue:
            anchor, before, after = queue.popleft()
            for point in expand_neighbors(
                client,
                collection_name,
                anchor,
                before=before,
                after=after,
            ):
                payload = point.payload if isinstance(point.payload, Mapping) else {}
                if payload.get("document_id") != seed.document_id:
                    continue
                neighbor_index = payload.get("chunk_index")
                if not isinstance(neighbor_index, int) or isinstance(neighbor_index, bool):
                    continue
                identity = _point_identity(point)
                if identity not in seed_identities:
                    retrieved[identity] = point
                if neighbor_index in visited_indexes:
                    continue
                visited_indexes.add(neighbor_index)
                candidate_pairs.append((seed, point, seed_reference, anchor))
                structure_type = detect_structure(payload)["type"]
                if (
                    structure_type in {"list", "literal", "numeral"}
                    and followed_list_chunks < max_list_continuation_chunks
                ):
                    followed_list_chunks += 1
                    queue.append(
                        (
                            _make_chunk(
                                point,
                                role="expanded",
                                seed_id=seed_reference,
                                reason="list_continuation_probe",
                            ),
                            1,
                            1,
                        )
                    )
        if max_list_continuation_chunks and followed_list_chunks >= max_list_continuation_chunks:
            warnings.append(
                f"Semilla {seed_reference}: se alcanzó el límite de "
                f"{max_list_continuation_chunks} chunks de continuación de listas."
            )
    timings["neighbor_expansion"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    accepted_chunks: list[ContextChunk] = []
    discarded: list[dict[str, Any]] = []
    for seed, point, seed_id, validation_anchor in candidate_pairs:
        payload = point.payload if isinstance(point.payload, Mapping) else {}
        decision = validate_neighbor(
            validation_anchor,
            dict(payload),
            query,
            require_same_document=require_same_document,
            require_consecutive=require_consecutive,
            complementary_terms=tuple(complementary_terms),
        )
        chunk_id = payload.get("chunk_id")
        if not decision.accepted:
            discarded.append(
                {
                    "chunk_id": chunk_id if isinstance(chunk_id, str) else None,
                    "document_id": payload.get("document_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "source_file": payload.get("source_file"),
                    "reason": decision.reason,
                    "seed_id": seed_id,
                }
            )
            continue
        chunk = _make_chunk(
            point,
            role=decision.role,
            seed_id=seed_id,
            reason=decision.reason,
        )
        accepted_chunks.append(chunk)
    timings["structural_validation"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    merged, duplicate_count, duplicate_records = deduplicate_and_sort(seeds + accepted_chunks)
    timings["deduplication_and_merging"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    final_chunks, budget_excluded, token_metadata = apply_token_budget(merged, options)
    for chunk in budget_excluded:
        discarded.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "source_file": chunk.source_file,
                "reason": "context_token_budget_exceeded",
                "seed_id": None,
            }
        )
    timings["token_budget"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    context = "\n\n".join(chunk.text.strip() for chunk in final_chunks if chunk.text.strip())
    provenance: dict[str, dict[str, list[Any]]] = {}
    for chunk in final_chunks:
        document = chunk.document_id or chunk.source_file or "unknown"
        entry = provenance.setdefault(document, {"source_files": [], "pages": []})
        if chunk.source_file and chunk.source_file not in entry["source_files"]:
            entry["source_files"].append(chunk.source_file)
        entry["pages"] = sorted(set(entry["pages"] + chunk.page_numbers))
    timings["context_assembly"] = time.perf_counter() - stage_start
    timings["total"] = time.perf_counter() - total_start

    neighbor_ids = {
        chunk_identity(chunk)
        for chunk in accepted_chunks
        if chunk_identity(chunk) not in seed_identities
    }
    metadata = {
        "metrics": {
            "seed_count": len(seeds),
            "neighbors_retrieved": len(retrieved),
            "neighbors_accepted": len(neighbor_ids),
            "neighbors_discarded": sum(
                1 for entry in discarded if entry["reason"] != "context_token_budget_exceeded"
            ),
            "complementary_chunks_included": sum(
                1 for chunk in final_chunks if chunk.role == "complementary"
            ),
            "duplicates_removed": duplicate_count,
            "chunks_excluded_by_budget": len(budget_excluded),
            "final_chunk_count": len(final_chunks),
            **token_metadata,
        },
        "discarded_chunks": discarded,
        "duplicate_chunks": duplicate_records,
        "warnings": warnings,
        "provenance": provenance,
        "included_chunk_ids": [chunk.chunk_id for chunk in final_chunks if chunk.chunk_id],
        "timings_seconds": timings,
    }
    return ContextResult(query=query, context=context, chunks=final_chunks, metadata=metadata)

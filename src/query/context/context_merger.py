"""Identidad de chunks, deduplicación, fusión de trazas y orden documental."""

from __future__ import annotations

from query.context.models import ContextChunk


def chunk_identity(chunk: ContextChunk) -> tuple[str, ...]:
    if chunk.chunk_id:
        return ("chunk_id", chunk.chunk_id)
    if chunk.document_id is not None and chunk.chunk_index is not None:
        return ("document_index", chunk.document_id, str(chunk.chunk_index))
    return ("content", chunk.document_id or "", chunk.source_file or "", chunk.text)


def deduplicate_and_sort(
    chunks: list[ContextChunk],
) -> tuple[list[ContextChunk], int, list[dict[str, object]]]:
    """Deduplica chunks y acumula en el retenido las trazas de inclusión.

    La identidad prioriza ``chunk_id``. Si falta, usa el par documento/índice
    y, finalmente, documento/origen/texto. Para duplicados, el chunk retenido
    acumula IDs de semilla y motivos distintos, conserva la mayor puntuación
    disponible y prioriza rol ``seed`` frente a ``complementary`` y
    ``expanded``. Cada ocurrencia duplicada eliminada queda registrada con
    identidad y semillas de origen.

    Al final, los chunks únicos se ordenan por documento, índice numérico
    (índices ausentes al final) e ID. El orden es determinista y documental;
    esta función no concatena ni reescribe el texto.

    Args:
        chunks: Semillas y vecinos aceptados candidatos.

    Returns:
        Tupla con chunks únicos ordenados, cantidad de ocurrencias eliminadas
        y registros de trazabilidad individuales de duplicados.
    """
    unique: dict[tuple[str, ...], ContextChunk] = {}
    duplicates = 0
    duplicate_records: list[dict[str, object]] = []
    for chunk in chunks:
        key = chunk_identity(chunk)
        existing = unique.get(key)
        if existing is None:
            unique[key] = chunk
            continue
        duplicates += 1
        duplicate_records.append(
            {
                "chunk_id": chunk.chunk_id,
                "duplicate_of_chunk_id": existing.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "reason": "duplicate_chunk_id" if chunk.chunk_id else "duplicate_fallback_identity",
                "included_by_seed_ids": list(chunk.included_by_seed_ids),
            }
        )
        existing.included_by_seed_ids = list(
            dict.fromkeys(existing.included_by_seed_ids + chunk.included_by_seed_ids)
        )
        existing.inclusion_reason = list(
            dict.fromkeys(existing.inclusion_reason + chunk.inclusion_reason)
        )
        if existing.reranker_score is None or (
            chunk.reranker_score is not None and chunk.reranker_score > existing.reranker_score
        ):
            existing.reranker_score = chunk.reranker_score
        if chunk.role == "seed":
            existing.role = "seed"
        elif chunk.role == "complementary" and existing.role != "seed":
            existing.role = "complementary"

    ordered = sorted(
        unique.values(),
        key=lambda chunk: (
            chunk.document_id or "",
            chunk.chunk_index if chunk.chunk_index is not None else float("inf"),
            chunk.chunk_id or "",
        ),
    )
    return ordered, duplicates, duplicate_records

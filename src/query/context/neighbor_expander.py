"""Recuperación acotada y de solo lectura de vecinos del mismo documento."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from query.context.models import ContextChunk


def expand_neighbors(
    client: QdrantClient,
    collection_name: str,
    seed: ContextChunk,
    before: int = 1,
    after: int = 1,
) -> list[Any]:
    """Recupera vecinos con payload mediante el ``scroll`` filtrado de Qdrant.

    La consulta combina una coincidencia exacta de ``document_id`` con el
    intervalo de ``chunk_index`` desde ``seed.chunk_index - before`` hasta
    ``seed.chunk_index + after``. Solicita payloads completos, omite vectores,
    limita el scroll a la ventana pedida, elimina el propio índice semilla y
    ordena los puntos restantes por índice. No recorre el corpus completo, no
    carga vectores ni cambia el estado de la colección.

    Args:
        client: Cliente Qdrant abierto por quien invoca esta función.
        collection_name: Colección existente que se consultará.
        seed: Semilla normalizada con identificador de documento e índice.
        before: Distancia máxima de índice a consultar antes de la semilla.
        after: Distancia máxima de índice a consultar después de la semilla.

    Returns:
        Puntos vecinos ordenados por ``chunk_index``. Devuelve una lista vacía
        si faltan datos de la semilla o si la ventana solicitada es vacía.

    Raises:
        ValueError: Si alguna cantidad de vecinos es negativa.
        Excepciones del cliente Qdrant: Se propagan si falla la lectura.
    """
    if before < 0 or after < 0:
        raise ValueError("La cantidad de vecinos no puede ser negativa.")
    if not seed.document_id or seed.chunk_index is None or before + after == 0:
        return []

    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=seed.document_id),
            ),
            models.FieldCondition(
                key="chunk_index",
                range=models.Range(
                    gte=seed.chunk_index - before,
                    lte=seed.chunk_index + after,
                ),
            ),
        ]
    )
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=query_filter,
        limit=before + after + 1,
        with_payload=True,
        with_vectors=False,
    )
    return sorted(
        (
            point
            for point in points
            if (point.payload or {}).get("chunk_index") != seed.chunk_index
        ),
        key=lambda point: (point.payload or {}).get("chunk_index", -1),
    )

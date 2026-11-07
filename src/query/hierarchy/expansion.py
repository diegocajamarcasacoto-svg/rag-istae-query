"""Recuperación jerárquica (arquitectura C): expansión hacia normas de rango superior.

Parte del contexto por artículos (modo ``hybrid``, arquitectura B) y le agrega
unidades de las normas superiores relacionadas con las normas recuperadas:

1. **Normas destino:** desde los ``document_id`` del contexto base se siguen las
   relaciones del catálogo (``Desarrolla`` y/o ``Debe ser compatible con``)
   solo hacia normas de **rango estrictamente superior** (número menor), hasta
   ``max_depth`` saltos. Las normas ya presentes en el contexto base no se
   vuelven a buscar.
2. **Búsqueda acotada:** la misma pregunta se busca (híbrido + rerank) filtrando
   por esas normas (``search``, inyectado por quien llama).
3. **Selección:** como máximo ``max_units`` unidades nuevas, ``max_per_document``
   por norma y, opcionalmente, con puntuación mínima ``min_score``. Se suben a
   artículos con ``build_unit_context`` y su propio presupuesto de tokens.

No se impulsa por rango: el rango decide **dónde** se busca, no el orden de los
resultados. La presentación final (orden por rango, regla del art. 425) es
responsabilidad de quien arma el prompt.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from query.context.models import ContextChunk, ContextResult
from query.hierarchy.catalog import RELATION_DEVELOPS, Catalog
from query.units import ChunkTextFetcher, UnitIndex, build_unit_context, estimate_tokens

HIERARCHY_SECTION_TITLE = "=== NORMAS DE RANGO SUPERIOR RELACIONADAS ==="

RankedResults = list[tuple[Any, float | None]]
HierarchySearch = Callable[[list[str]], RankedResults]


@dataclass(frozen=True)
class HierarchyTarget:
    """Norma superior alcanzada desde el contexto base."""

    document_id: str
    rank: int
    depth: int
    via: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (norma origen, relación)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "rank": self.rank,
            "depth": self.depth,
            "via": [{"source_document_id": source, "relation": relation} for source, relation in self.via],
        }


def hierarchy_targets(
    catalog: Catalog,
    seed_document_ids: Iterable[str],
    relations: Sequence[str] = (RELATION_DEVELOPS,),
    max_depth: int = 2,
) -> tuple[list[HierarchyTarget], list[str]]:
    """Normas de rango superior alcanzables desde las semillas (BFS, sin ciclos).

    Devuelve ``(destinos, advertencias)``. Los destinos se ordenan por rango
    (mayor jerarquía primero) y luego por ``document_id``.
    """
    if max_depth < 1:
        raise ValueError("max_depth debe ser al menos 1.")
    seeds = list(dict.fromkeys(seed_document_ids))
    warnings = [f"{doc} no figura en el catálogo: no se expande." for doc in seeds if catalog.document(doc) is None]
    seed_set = set(seeds)

    found: dict[str, dict[str, Any]] = {}
    queue = deque((doc, 0) for doc in seeds if catalog.document(doc) is not None)
    visited = {doc for doc, _ in queue}
    while queue:
        source, depth = queue.popleft()
        if depth >= max_depth:
            continue
        source_rank = catalog.documents[source].rank
        for rel in catalog.outgoing(source, relations):
            target = catalog.document(rel.target_document_id)
            if target is None or target.rank >= source_rank:
                continue  # solo hacia rango estrictamente superior
            if target.document_id in seed_set:
                continue
            entry = found.get(target.document_id)
            if entry is None:
                found[target.document_id] = {"depth": depth + 1, "via": [(source, rel.relation)]}
            elif entry["depth"] == depth + 1 and (source, rel.relation) not in entry["via"]:
                entry["via"].append((source, rel.relation))
            if target.document_id not in visited:
                visited.add(target.document_id)
                queue.append((target.document_id, depth + 1))

    targets = [
        HierarchyTarget(doc, catalog.documents[doc].rank, info["depth"], tuple(info["via"]))
        for doc, info in found.items()
    ]
    targets.sort(key=lambda target: (target.rank, target.document_id))
    return targets, warnings


def _chunk_id(point: Any) -> str | None:
    payload = getattr(point, "payload", None)
    value = payload.get("chunk_id") if isinstance(payload, dict) else None
    return value if isinstance(value, str) else None


def _document_id(point: Any) -> str:
    payload = getattr(point, "payload", None)
    return str(payload.get("document_id") or "") if isinstance(payload, dict) else ""


def select_hierarchy_results(
    ranked: RankedResults,
    index: UnitIndex,
    *,
    exclude_unit_ids: Iterable[str] = (),
    max_units: int = 3,
    max_per_document: int = 1,
    min_score: float | None = None,
) -> tuple[RankedResults, list[dict[str, Any]]]:
    """Elige los resultados de las unidades nuevas respetando los topes.

    Conserva todos los chunks acertados de cada unidad elegida (para el envío
    parcial de unidades largas). Devuelve ``(resultados, descartes)``.
    """
    excluded = set(exclude_unit_ids)
    chosen: set[str] = set()
    per_document: dict[str, int] = {}
    selected: RankedResults = []
    skipped: list[dict[str, Any]] = []
    skipped_units: set[str] = set()
    for point, score in ranked:
        chunk_id = _chunk_id(point)
        if chunk_id is None:
            continue
        unit = index.unit_for_chunk(chunk_id)
        unit_id = unit.unit_id if unit else chunk_id
        document_id = unit.document_id if unit else _document_id(point)
        if unit_id in chosen:
            selected.append((point, score))
            continue
        reason = None
        if unit_id in excluded:
            reason = "already_in_base_context"
        elif min_score is not None and score is not None and float(score) < min_score:
            reason = "below_min_score"
        elif len(chosen) >= max_units:
            reason = "max_units"
        elif per_document.get(document_id, 0) >= max_per_document:
            reason = "max_per_document"
        if reason:
            if unit_id not in skipped_units:  # una entrada por unidad (su mejor chunk)
                skipped_units.add(unit_id)
                skipped.append({"unit_id": unit_id, "chunk_id": chunk_id, "reason": reason,
                                "score": float(score) if score is not None else None})
            continue
        chosen.add(unit_id)
        per_document[document_id] = per_document.get(document_id, 0) + 1
        selected.append((point, score))
    return selected, skipped


def _with_rank(chunk: ContextChunk, catalog: Catalog) -> ContextChunk:
    """Agrega el rango jerárquico a la línea de encabezado del bloque."""
    document = catalog.document(chunk.document_id or "")
    if document is None:
        return chunk
    header, _, body = chunk.text.partition("\n")
    header = f"{header} · Jerarquía {document.rank}: {document.rank_label}"
    return replace(chunk, text=f"{header}\n{body}" if body else header)


def expand_with_hierarchy(
    query: str,
    base: ContextResult,
    index: UnitIndex,
    catalog: Catalog,
    search: HierarchySearch,
    *,
    relations: Sequence[str] = (RELATION_DEVELOPS,),
    max_depth: int = 2,
    max_units: int = 3,
    max_per_document: int = 1,
    min_score: float | None = None,
    max_tokens: int | None = 1500,
    max_unit_tokens: int = 1500,
    fetch_chunk_texts: ChunkTextFetcher | None = None,
) -> ContextResult:
    """Contexto ``hybrid`` (por artículos) + unidades de normas superiores relacionadas."""
    if max_units < 1 or max_per_document < 1:
        raise ValueError("max_units y max_per_document deben ser positivos.")
    start = time.perf_counter()
    seed_documents = [chunk.document_id for chunk in base.chunks if chunk.document_id]
    targets, warnings = hierarchy_targets(catalog, seed_documents, relations, max_depth)
    target_ids = [target.document_id for target in targets]
    via_by_document = {target.document_id: target for target in targets}

    extra_chunks: list[ContextChunk] = []
    ranked: RankedResults = []
    skipped: list[dict[str, Any]] = []
    extra_metadata: dict[str, Any] = {}
    search_seconds = None
    if target_ids:
        search_start = time.perf_counter()
        ranked = search(target_ids)
        search_seconds = time.perf_counter() - search_start
        selected, skipped = select_hierarchy_results(
            ranked,
            index,
            exclude_unit_ids=[chunk.chunk_id for chunk in base.chunks if chunk.chunk_id],
            max_units=max_units,
            max_per_document=max_per_document,
            min_score=min_score,
        )
        if selected:
            extra = build_unit_context(
                query, selected, index,
                fetch_chunk_texts=fetch_chunk_texts,
                max_tokens=max_tokens,
                max_unit_tokens=max_unit_tokens,
            )
            extra_metadata = extra.metadata
            warnings.extend(extra.metadata.get("warnings") or [])
            for chunk in extra.chunks:
                extra_chunks.append(replace(
                    _with_rank(chunk, catalog),
                    role="hierarchy",
                    inclusion_reason=[*chunk.inclusion_reason, "hierarchy"],
                ))
    else:
        warnings.append("Sin normas superiores relacionadas con el contexto base.")

    base_chunks = [_with_rank(chunk, catalog) for chunk in base.chunks]
    context = "\n\n".join(chunk.text for chunk in base_chunks)
    if extra_chunks:
        context += "\n\n" + HIERARCHY_SECTION_TITLE + "\n" + "\n\n".join(chunk.text for chunk in extra_chunks)

    base_metrics = dict(base.metadata.get("metrics") or {})
    metadata = {
        **base.metadata,
        "retrieval_mode": "hierarchical",
        "metrics": {
            **base_metrics,
            "base_units": len(base_chunks),
            "hierarchy_units": len(extra_chunks),
            "final_chunk_count": len(base_chunks) + len(extra_chunks),
            "base_token_count": base_metrics.get("token_count"),
            "token_count": estimate_tokens(context),
            "token_count_is_estimated": True,
        },
        "included_chunk_ids": [chunk.chunk_id for chunk in base_chunks + extra_chunks],
        "warnings": [*(base.metadata.get("warnings") or []), *warnings],
        "hierarchy": {
            "relations": list(relations),
            "max_depth": max_depth,
            "max_units": max_units,
            "max_per_document": max_per_document,
            "min_score": min_score,
            "token_budget": max_tokens,
            "seed_documents": list(dict.fromkeys(seed_documents)),
            "targets": [target.to_dict() for target in targets],
            "candidates": len(ranked),
            "units": [
                {
                    "unit_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "rank": catalog.documents[chunk.document_id].rank
                    if chunk.document_id in catalog.documents else None,
                    "score": chunk.reranker_score,
                    "matched_chunk_ids": chunk.included_by_seed_ids,
                    "mode": chunk.inclusion_reason[0],
                    "via": via_by_document[chunk.document_id].to_dict()["via"]
                    if chunk.document_id in via_by_document else [],
                }
                for chunk in extra_chunks
            ],
            "skipped": skipped,
            "excluded_units": extra_metadata.get("excluded_units") or [],
        },
        "timings_seconds": {
            **(base.metadata.get("timings_seconds") or {}),
            "hierarchy_search": search_seconds,
            "hierarchy_total": time.perf_counter() - start,
        },
    }
    return ContextResult(query=query, context=context, chunks=base_chunks + extra_chunks, metadata=metadata)

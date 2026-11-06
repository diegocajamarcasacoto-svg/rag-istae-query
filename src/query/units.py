"""Expansión por artículo (*small-to-big*): de los chunks recuperados a sus unidades normativas.

La recuperación híbrida y el reranking siguen operando sobre los chunks finos
de Docling. Esta etapa reemplaza la expansión heurística de vecinos: cada
chunk recuperado se sube a la unidad normativa (artículo, disposición,
sección o preámbulo) a la que pertenece según ``units.jsonl``, generado
offline por ``rag_istae_ingestion`` (``build_units.py`` o ``run_pipeline.py``).

No carga modelos, no ejecuta búsquedas y no escribe en Qdrant. Solo lee
textos de chunks (``fetch_chunk_texts``) cuando una unidad es demasiado larga
y se envían únicamente sus partes pertinentes.

El resultado es un ``ContextResult`` compatible con la generación. En este
modo cada elemento de ``ContextResult.chunks`` representa una **unidad**:
``chunk_id`` contiene el ``unit_id`` (p. ej. ``INT08-art12``) y
``included_by_seed_ids`` los chunks recuperados que la seleccionaron.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import models

from query.context.models import ContextChunk, ContextResult

PARTIAL_SEPARATOR = "[…]"

ChunkTextFetcher = Callable[[list[str]], dict[str, str]]


@dataclass(frozen=True)
class Unit:
    unit_id: str
    document_id: str
    kind: str
    label: str
    title: str
    path: str
    chunk_ids: tuple[str, ...]
    text: str


class UnitIndex:
    """Índice en memoria ``chunk_id → unidad`` construido desde ``units.jsonl``."""

    def __init__(self, units: Sequence[Unit]) -> None:
        self._units: dict[str, Unit] = {}
        self._by_chunk: dict[str, Unit] = {}
        for unit in units:
            if unit.unit_id in self._units:
                raise ValueError(f"unit_id duplicado en units.jsonl: {unit.unit_id}")
            self._units[unit.unit_id] = unit
            for chunk_id in unit.chunk_ids:
                if chunk_id in self._by_chunk:
                    raise ValueError(f"El chunk {chunk_id} pertenece a más de una unidad.")
                self._by_chunk[chunk_id] = unit

    def __len__(self) -> int:
        return len(self._units)

    def unit(self, unit_id: str) -> Unit | None:
        return self._units.get(unit_id)

    def unit_for_chunk(self, chunk_id: str) -> Unit | None:
        return self._by_chunk.get(chunk_id)

    @property
    def chunk_count(self) -> int:
        return len(self._by_chunk)


def load_units(path: Path) -> UnitIndex:
    """Lee ``units.jsonl`` y valida su estructura mínima."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Genérelo en rag_istae_ingestion con scripts/build_units.py."
        )
    units = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                units.append(Unit(
                    unit_id=row["unit_id"],
                    document_id=row["document_id"],
                    kind=row.get("kind") or "",
                    label=row.get("label") or "",
                    title=row.get("title") or "",
                    path=row.get("path") or "",
                    chunk_ids=tuple(row.get("chunk_ids") or ()),
                    text=row.get("text") or "",
                ))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Línea {line_no} inválida en {path}: {exc}") from exc
    if not units:
        raise ValueError(f"{path} no contiene unidades.")
    return UnitIndex(units)


def estimate_tokens(text: str) -> int:
    """Misma estimación que el Context Builder: bytes UTF-8 / 4, hacia arriba."""
    return math.ceil(len(text.encode("utf-8")) / 4) if text else 0


def qdrant_chunk_fetcher(client: Any, collection_name: str) -> ChunkTextFetcher:
    """Lee los textos de chunks por ``chunk_id`` (solo lectura, filtro sobre el payload)."""

    def fetch(chunk_ids: list[str]) -> dict[str, str]:
        if not chunk_ids:
            return {}
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="chunk_id", match=models.MatchAny(any=list(chunk_ids)))]
            ),
            limit=len(chunk_ids),
            with_payload=True,
            with_vectors=False,
        )
        texts = {}
        for point in points:
            payload = point.payload or {}
            if isinstance(payload.get("chunk_id"), str):
                texts[payload["chunk_id"]] = str(payload.get("text") or "")
        return texts

    return fetch


def _payload(point: Any) -> dict[str, Any]:
    payload = getattr(point, "payload", None)
    return payload if isinstance(payload, dict) else {}


def _score(point: Any, score: float | None) -> float | None:
    value = score if score is not None else getattr(point, "score", None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _partial_chunk_ids(unit: Unit, matched: list[str], window: int) -> list[list[str]]:
    """Tramos contiguos (dentro de la unidad) alrededor de los chunks acertados."""
    positions = {unit.chunk_ids.index(chunk_id) for chunk_id in matched if chunk_id in unit.chunk_ids}
    selected = sorted({
        index
        for position in positions
        for index in range(max(0, position - window), min(len(unit.chunk_ids), position + window + 1))
    })
    spans: list[list[str]] = []
    for index in selected:
        if spans and index == unit.chunk_ids.index(spans[-1][-1]) + 1:
            spans[-1].append(unit.chunk_ids[index])
        else:
            spans.append([unit.chunk_ids[index]])
    return spans


def _header(unit: Unit) -> str:
    parts = [unit.document_id, unit.label]
    if unit.title:
        parts.append(unit.title)
    return f"[{unit.unit_id}] " + " · ".join(part for part in parts if part)


def build_unit_context(
    query: str,
    ranked_results: list[tuple[Any, float | None]],
    index: UnitIndex,
    *,
    fetch_chunk_texts: ChunkTextFetcher | None = None,
    max_tokens: int | None = 4000,
    max_unit_tokens: int = 1500,
    partial_window: int = 1,
) -> ContextResult:
    """Construye el contexto con las unidades de los chunks recuperados.

    - Las unidades se ordenan por el mejor puesto de sus chunks en
      ``ranked_results`` y se deduplican por ``unit_id``.
    - Una unidad de más de ``max_unit_tokens`` se envía parcialmente: sus
      chunks acertados con ``partial_window`` vecinos **dentro de la misma
      unidad**, separando tramos con ``[…]``. Requiere ``fetch_chunk_texts``.
    - ``max_tokens`` limita el contexto total: la primera unidad siempre se
      incluye; las siguientes solo si caben. ``None`` desactiva el límite.
    - Un chunk que no figura en el índice (``units.jsonl`` desactualizado) se
      incluye solo, con una advertencia.
    """
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens debe ser positivo o None.")
    if max_unit_tokens < 1 or partial_window < 0:
        raise ValueError("max_unit_tokens debe ser positivo y partial_window no negativo.")

    start = time.perf_counter()
    warnings: list[str] = []
    groups: dict[str, dict[str, Any]] = {}
    for rank, (point, score) in enumerate(ranked_results, start=1):
        payload = _payload(point)
        chunk_id = payload.get("chunk_id")
        if not isinstance(chunk_id, str):
            warnings.append(f"Resultado #{rank} sin chunk_id: se omite.")
            continue
        unit = index.unit_for_chunk(chunk_id)
        if unit is None:
            warnings.append(f"{chunk_id} no figura en units.jsonl: se incluye como chunk suelto.")
            unit = Unit(chunk_id, str(payload.get("document_id") or ""), "chunk", "", "", "", (chunk_id,),
                        str(payload.get("text") or ""))
        group = groups.setdefault(unit.unit_id, {
            "unit": unit,
            "best_rank": rank,
            "best_score": _score(point, score),
            "matched": [],
            "source_files": [],
            "pages": set(),
            "headings": list(payload.get("headings") or []),
        })
        group["matched"].append(chunk_id)
        source_file = payload.get("source_file")
        if isinstance(source_file, str) and source_file not in group["source_files"]:
            group["source_files"].append(source_file)
        group["pages"].update(p for p in payload.get("page_numbers") or [] if isinstance(p, int))

    ordered = sorted(groups.values(), key=lambda group: group["best_rank"])

    # Textos de las unidades demasiado largas: solo los tramos pertinentes.
    partial_spans: dict[str, list[list[str]]] = {}
    for group in ordered:
        unit = group["unit"]
        if estimate_tokens(unit.text) > max_unit_tokens and len(unit.chunk_ids) > 1:
            partial_spans[unit.unit_id] = _partial_chunk_ids(unit, group["matched"], partial_window)
    fetched: dict[str, str] = {}
    if partial_spans:
        if fetch_chunk_texts is None:
            warnings.append("Unidades largas enviadas completas: no se indicó fetch_chunk_texts.")
            partial_spans = {}
        else:
            needed = [chunk_id for spans in partial_spans.values() for span in spans for chunk_id in span]
            fetched = fetch_chunk_texts(needed)

    included: list[ContextChunk] = []
    excluded: list[dict[str, Any]] = []
    blocks: list[str] = []
    used_tokens = 0
    for group in ordered:
        unit = group["unit"]
        spans = partial_spans.get(unit.unit_id)
        if spans:
            missing = [chunk_id for span in spans for chunk_id in span if chunk_id not in fetched]
            if missing:
                warnings.append(f"{unit.unit_id}: no se pudieron leer {len(missing)} chunks.")
            body = f"\n\n{PARTIAL_SEPARATOR}\n\n".join(
                "\n\n".join(fetched[chunk_id] for chunk_id in span if chunk_id in fetched) for span in spans
            )
            reason = "unit_partial"
        else:
            body = unit.text
            reason = "unit_full"
        block = f"{_header(unit)}\n{body}".strip()
        block_tokens = estimate_tokens(block)
        if included and max_tokens is not None and used_tokens + block_tokens > max_tokens:
            excluded.append({"unit_id": unit.unit_id, "reason": "context_token_budget_exceeded",
                             "tokens": block_tokens})
            continue
        blocks.append(block)
        used_tokens += block_tokens
        included.append(ContextChunk(
            chunk_id=unit.unit_id,
            document_id=unit.document_id or None,
            chunk_index=None,
            source_file=group["source_files"][0] if group["source_files"] else None,
            page_numbers=sorted(group["pages"]),
            headings=group["headings"],
            item_types=[],
            text=block,
            reranker_score=group["best_score"],
            role="seed",
            structural_type=unit.kind or "unknown",
            structural_id=unit.unit_id,
            structure_status="unit",
            included_by_seed_ids=list(group["matched"]),
            inclusion_reason=[reason],
        ))

    context = "\n\n".join(blocks)
    token_count = estimate_tokens(context)
    metadata = {
        "expansion": "units",
        "metrics": {
            "ranked_chunks": len(ranked_results),
            "units_selected": len(ordered),
            "units_included": len(included),
            "units_partial": sum(1 for chunk in included if chunk.inclusion_reason == ["unit_partial"]),
            "units_excluded_by_budget": len(excluded),
            "final_chunk_count": len(included),
            "token_count": token_count,
            "token_count_is_estimated": True,
            "token_budget": max_tokens,
            "max_unit_tokens": max_unit_tokens,
        },
        "units": [
            {
                "unit_id": chunk.chunk_id,
                "matched_chunk_ids": chunk.included_by_seed_ids,
                "mode": chunk.inclusion_reason[0],
                "score": chunk.reranker_score,
            }
            for chunk in included
        ],
        "excluded_units": excluded,
        "warnings": warnings,
        "included_chunk_ids": [chunk.chunk_id for chunk in included],
        "timings_seconds": {"total": time.perf_counter() - start},
    }
    return ContextResult(query=query, context=context, chunks=included, metadata=metadata)

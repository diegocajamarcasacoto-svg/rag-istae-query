"""Motor de recuperación residente: carga los recursos una vez y atiende consultas.

Lo usa el servicio ``/retrieve``. Ejecuta el mismo pipeline que
``scripts/query_rag.py`` (``query.pipeline.run_units_pipeline``) con los
parámetros de la línea base (``query.baseline.CURRENT_BASELINE``). Por consulta
solo se eligen ``rerank``, ``hierarchy`` (B/C) y ``document_ids``.

Qdrant local admite un solo proceso y la inferencia es en CPU: las consultas se
atienden de a una (``threading.Lock``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from query.baseline import CURRENT_BASELINE, Baseline
from query.context.models import ContextResult
from query.hierarchy import load_catalog
from query.hierarchy.catalog import Catalog, file_sha256
from query.pipeline import run_units_pipeline
from query.presentation import unit_body
from query.reranking import load_reranker
from query.retrieval import load_e5, open_client
from query.units import UnitIndex, load_units


@dataclass
class RetrievalOutput:
    """Resultado de una consulta, listo para serializar."""

    units: list[dict[str, Any]]
    context: str
    tokens_estimated: int | None
    warnings: list[str]
    timings: dict[str, float | None]
    debug: dict[str, Any] = field(default_factory=dict)
    context_result: ContextResult | None = None


def _path(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not value:
        raise KeyError(f"Falta {key} en la configuración del servicio.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No existe {key}: {path}")
    return path


def describe_units(result: ContextResult, index: UnitIndex, catalog: Catalog | None) -> list[dict[str, Any]]:
    """Artículos del contexto en el orden en que se envían (base y luego jerarquía)."""
    units = []
    for chunk in result.chunks:
        unit = index.unit(chunk.chunk_id or "")
        info = catalog.document(chunk.document_id or "") if catalog else None
        units.append({
            "unit_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": info.name if info else chunk.document_id,
            "rank": info.rank if info else None,
            "label": unit.label if unit else "",
            "title": unit.title if unit else "",
            "text": unit_body(chunk),
            "partial": "unit_partial" in chunk.inclusion_reason,
            "score": chunk.reranker_score,
            "origin": "hierarchy" if chunk.role == "hierarchy" else "base",
            "matched_chunk_ids": list(chunk.included_by_seed_ids),
        })
    return units


def _candidates(points: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {"chunk_id": (point.payload or {}).get("chunk_id"), "rrf_score": getattr(point, "score", None)}
        for point in points
    ]


class RetrievalEngine:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        e5: Any,
        reranker: Any,
        client: Any,
        unit_index: UnitIndex,
        catalog: Catalog,
        params: Baseline = CURRENT_BASELINE,
        index_version: dict[str, str] | None = None,
        load_timings: dict[str, float] | None = None,
    ) -> None:
        self.config = config
        self.e5 = e5
        self.reranker = reranker
        self.client = client
        self.unit_index = unit_index
        self.catalog = catalog
        self.params = params
        self.index_version = index_version or {}
        self.load_timings = load_timings or {}
        self._lock = threading.Lock()
        self.document_ids = frozenset(catalog.documents) | unit_index.document_ids

    @classmethod
    def from_config(cls, config: dict[str, Any], params: Baseline = CURRENT_BASELINE) -> RetrievalEngine:
        """Carga E5, el reranker, ``units.jsonl``, el catálogo y abre Qdrant (solo lectura)."""
        units_path = _path(config, "units_path")
        catalog_path = _path(config, "catalog_path")
        timings: dict[str, float] = {}

        def timed(stage: str, load):
            start = time.perf_counter()
            value = load()
            timings[stage] = time.perf_counter() - start
            return value

        e5 = timed("e5_load", lambda: load_e5(config))
        reranker = timed("reranker_load", lambda: load_reranker(config["reranker_model_path"]))
        unit_index = timed("units_load", lambda: load_units(units_path))
        catalog = timed("catalog_load", lambda: load_catalog(catalog_path))
        client = timed("qdrant_open", lambda: open_client(config))
        return cls(
            config=config, e5=e5, reranker=reranker, client=client,
            unit_index=unit_index, catalog=catalog, params=params,
            index_version={
                "collection": str(config["collection_name"]),
                "units_sha256": file_sha256(units_path),
                "catalog_sha256": file_sha256(catalog_path),
            },
            load_timings=timings,
        )

    def info(self) -> dict[str, Any]:
        return {
            "baseline": self.params.id,
            "units": len(self.unit_index),
            "documents": len(self.document_ids),
            "index_version": self.index_version,
            "load_seconds": self.load_timings,
        }

    def retrieve(
        self,
        question: str,
        *,
        rerank: bool = True,
        hierarchy: bool = False,
        document_ids: Sequence[str] | None = None,
    ) -> RetrievalOutput:
        state: dict[str, Any] = {"timings": {}}
        start = time.perf_counter()
        with self._lock:
            result = run_units_pipeline(
                question,
                e5=self.e5,
                reranker=self.reranker if rerank else None,
                client=self.client,
                config=self.config,
                unit_index=self.unit_index,
                catalog=self.catalog,
                params=self.params,
                rerank=rerank,
                hierarchy=hierarchy,
                document_ids=document_ids,
                state=state,
            )
        timings = dict(result.timings)
        timings["total"] = time.perf_counter() - start
        context_result = result.context_result
        metadata = context_result.metadata
        debug = {
            "candidates": _candidates(result.candidates),
            "ranked": [
                {"chunk_id": (point.payload or {}).get("chunk_id"),
                 "rerank_score": float(score) if score is not None else None}
                for point, score in result.ranked_results
            ],
            "excluded_units": metadata.get("excluded_units") or [],
            "hierarchy": metadata.get("hierarchy"),
            "document_order": metadata.get("document_order"),
        }
        return RetrievalOutput(
            units=describe_units(context_result, self.unit_index, self.catalog),
            context=context_result.context,
            tokens_estimated=(metadata.get("metrics") or {}).get("token_count"),
            warnings=list(metadata.get("warnings") or []),
            timings=timings,
            debug=debug,
            context_result=context_result,
        )

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001 - el cierre no debe ocultar otro error
            pass

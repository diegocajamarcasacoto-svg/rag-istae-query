"""Parámetros de la línea base B/C en un solo lugar.

LB-1 (2026-11-06) se describe en la sección «Línea base B/C» del README y su
justificación en ``Justificacion_Linea_Base.md``; ``candidate_k`` = 20 está
ratificado por la Prueba 5 (``Resultados_Prueba5_Candidatos.md``).

Los usan los valores por defecto de ``scripts/query_rag.py`` y el servicio
``/retrieve``, que **no** los expone en el request. ``tests/test_baseline.py``
los fija: cambiarlos debe ser una decisión explícita (una nueva versión).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from query.hierarchy.catalog import RELATION_COMPATIBLE, RELATION_DEVELOPS


@dataclass(frozen=True)
class Baseline:
    id: str
    # Recuperación híbrida y reranking.
    top_k: int
    candidate_k: int
    prefetch_k: int
    rrf_k: int
    rerank_batch_size: int
    # Contexto por artículos (--expansion units).
    max_context_tokens: int
    max_unit_tokens: int
    # Capa jerárquica (arquitectura C).
    hierarchy_relations: tuple[str, ...]
    hierarchy_depth: int
    hierarchy_search: str
    hierarchy_candidate_k: int
    hierarchy_max_units: int
    hierarchy_max_per_document: int
    hierarchy_max_tokens: int
    hierarchy_min_score: float | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hierarchy_relations"] = list(self.hierarchy_relations)
        return data


LB1 = Baseline(
    id="LB-1",
    top_k=5,
    candidate_k=20,
    prefetch_k=20,
    rrf_k=60,
    rerank_batch_size=4,
    max_context_tokens=4000,
    max_unit_tokens=1500,
    hierarchy_relations=(RELATION_DEVELOPS, RELATION_COMPATIBLE),
    hierarchy_depth=2,
    hierarchy_search="joint",
    hierarchy_candidate_k=20,
    hierarchy_max_units=3,
    hierarchy_max_per_document=1,
    hierarchy_max_tokens=1500,
    hierarchy_min_score=None,
)

CURRENT_BASELINE = LB1

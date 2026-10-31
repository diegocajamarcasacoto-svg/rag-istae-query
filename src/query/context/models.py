"""Estructuras tipadas compartidas entre las etapas de construcción de contexto."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ChunkRole = Literal["seed", "expanded", "complementary"]


@dataclass
class ContextChunk:
    """Representa un chunk normalizado recuperado o añadido por expansión.

    La metadata de los payloads Qdrant es opcional: los identificadores y datos
    de procedencia ausentes se mantienen como ``None`` o listas vacías, sin
    inventar valores. ``structural_type`` y ``structure_status`` conservan la
    clasificación heurística por separado de ``headings`` e ``item_types``.
    Las semillas mantienen su puntuación de reranker; si no se ejecutó el
    reranker, el valor es ``None``.
    """

    chunk_id: str | None
    document_id: str | None
    chunk_index: int | None
    source_file: str | None
    page_numbers: list[int]
    headings: list[str]
    item_types: list[str]
    text: str
    reranker_score: float | None = None
    role: ChunkRole = "expanded"
    structural_type: str = "unknown"
    structural_id: str | None = None
    structure_status: str = "unknown"
    included_by_seed_ids: list[str] = field(default_factory=list)
    inclusion_reason: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Devuelve los campos del dataclass como un diccionario recursivo."""
        return asdict(self)


@dataclass
class ContextResult:
    """Contexto consolidado, chunks ordenados y metadata de trazabilidad.

    ``metadata`` contiene métricas funcionales, tiempos por etapa, procedencia,
    candidatos descartados, duplicados, advertencias y estado del presupuesto.
    El texto final se genera a partir de los chunks incluidos y ordenados.
    """

    query: str
    context: str
    chunks: list[ContextChunk]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serializa el resultado y convierte cada chunk a un diccionario."""
        return {
            "query": self.query,
            "context": self.context,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metadata": self.metadata,
        }

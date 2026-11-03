"""Tipos del contrato de datos del corpus normativo.

Estos tipos no dependen de Qdrant, modelos de embeddings ni del proveedor LLM,
para que la variante de contexto largo pueda ejecutarse sin esas dependencias
y para que luego puedan moverse a un módulo ``core`` compartido.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

UnitKind = Literal["preambulo", "articulo", "seccion", "disposicion"]
StructureMode = Literal["articulado", "secciones"]


@dataclass(frozen=True)
class DocumentInfo:
    """Ficha de una norma del corpus, tomada del catálogo normativo."""

    document_id: str
    code: str
    name: str
    doc_type: str
    issuer: str
    domain: str
    rank: int
    rank_label: str
    source_file: str


@dataclass(frozen=True)
class Relation:
    """Relación entre dos normas (por ejemplo, "Desarrolla")."""

    source_document_id: str | None
    source_code: str
    relation: str
    target_document_id: str | None
    target_code: str
    verification: str = ""


@dataclass
class Unit:
    """Unidad citable de una norma: artículo, sección, disposición o preámbulo.

    ``unit_id`` es estable y legible (``INT08-art12``): depende de la norma y de
    su numeración, no de la posición del texto ni de la herramienta de
    extracción.
    """

    unit_id: str
    kind: UnitKind
    label: str
    title: str
    path: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocument:
    """Documento estructurado en unidades, con diagnóstico del análisis."""

    document_id: str
    mode: StructureMode
    units: list[Unit]
    warnings: list[str] = field(default_factory=list)
    rejected_article_candidates: list[str] = field(default_factory=list)

    def count(self, kind: UnitKind) -> int:
        return sum(1 for unit in self.units if unit.kind == kind)

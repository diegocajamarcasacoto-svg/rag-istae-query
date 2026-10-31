"""Reglas de decisión que separan detección estructural e inclusión."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from query.context.models import ContextChunk
from query.context.structure_detector import detect_structure, normalize_text


DEFAULT_COMPLEMENTARY_TERMS = (
    "concordancia",
    "concordancias",
    "reforma",
    "reformatoria",
    "reformatorias",
    "reformado",
    "reformada",
    "vigencia",
    "modificacion",
    "modificada",
    "modificado",
    "nota",
)


@dataclass(frozen=True)
class ValidationDecision:
    """Resultado y motivo auditable de validar un vecino candidato."""

    accepted: bool
    reason: str
    role: str
    structure: dict[str, str | None]


def is_complementary(structure_type: str) -> bool:
    return structure_type in {"reform_note", "concordance", "complementary"}


def validate_neighbor(
    seed: ContextChunk,
    neighbor_payload: dict[str, Any] | None,
    query: str,
    *,
    require_same_document: bool = True,
    require_consecutive: bool = True,
    complementary_terms: tuple[str, ...] = DEFAULT_COMPLEMENTARY_TERMS,
) -> ValidationDecision:
    """Decide si un vecino candidato puede ampliar la ventana de una semilla.

    Primero detecta la estructura del candidato y después aplica las
    restricciones de documento e índice. Un artículo, capítulo o sección
    identificado con tipo o ID diferente constituye un límite explícito y se
    rechaza; no se presupone que continúe la semilla. Las notas de reforma,
    concordancias y notas genéricas son condicionales: la consulta debe contener
    al menos uno de los términos configurados, normalizado. Listas, literales,
    numerales y estructuras desconocidas se admiten si cumplen la proximidad.
    Cada decisión incluye un motivo estable para trazabilidad.

    La comprobación de términos es léxica y explícita: normaliza mayúsculas y
    acentos, pero no infiere intención semántica.

    Args:
        seed: Semilla o ancla desde la que se calcula la proximidad.
        neighbor_payload: Payload Qdrant original del candidato.
        query: Consulta original que controla contenido complementario.
        require_same_document: Rechaza documentos ausentes o distintos.
        require_consecutive: Exige una diferencia de índice de exactamente
            uno respecto de la semilla/ancla recibida.
        complementary_terms: Señales textuales que pueden justificar
            material complementario.

    Returns:
        ``ValidationDecision`` con aceptación, motivo, rol asignado y
        estructura detectada.
    """
    payload = neighbor_payload or {}
    structure = detect_structure(payload)
    candidate_doc = payload.get("document_id")
    candidate_index = payload.get("chunk_index")

    if require_same_document and (
        not seed.document_id or candidate_doc != seed.document_id
    ):
        return ValidationDecision(False, "different_or_missing_document", "expanded", structure)
    if require_consecutive and (
        seed.chunk_index is None
        or not isinstance(candidate_index, int)
        or isinstance(candidate_index, bool)
        or abs(candidate_index - seed.chunk_index) != 1
    ):
        return ValidationDecision(False, "not_consecutive", "expanded", structure)

    seed_type = seed.structural_type
    candidate_type = str(structure["type"])
    candidate_id = structure["id"]
    if candidate_type in {"article", "chapter", "section"} and (
        candidate_type != seed_type or candidate_id != seed.structural_id
    ):
        return ValidationDecision(False, "explicit_structural_boundary", "expanded", structure)

    if is_complementary(candidate_type):
        normalized_query = normalize_text(query)
        normalized_terms = tuple(normalize_text(term) for term in complementary_terms)
        if not any(term and term in normalized_query for term in normalized_terms):
            return ValidationDecision(
                False,
                "conditional_complement_not_justified_by_query",
                "complementary",
                structure,
            )
        return ValidationDecision(True, "query_justifies_complementary_content", "complementary", structure)

    if candidate_type in {"list", "literal", "numeral"}:
        reason = "consecutive_list_or_subitem"
    elif candidate_type == "unknown":
        reason = "consecutive_structure_unknown"
    elif seed_type == candidate_type and seed.structural_id == candidate_id:
        reason = "same_structural_unit"
    else:
        reason = "consecutive_document_neighbor"
    return ValidationDecision(True, reason, "expanded", structure)

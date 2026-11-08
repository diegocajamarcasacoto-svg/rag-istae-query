"""Contrato de ``POST /retrieve`` (ver ``docs/contrato_retrieve.md``).

El request solo elige qué arquitectura se ejecuta (``rerank``, ``hierarchy``) y,
opcionalmente, a qué normas se limita la búsqueda. Los parámetros de la línea
base (top_k, candidatos, RRF, presupuestos, topes de jerarquía) los fija el
servidor: cualquier campo desconocido, como ``top_k`` o ``history``, da 422.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONTRACT_VERSION = "1.0"


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="Consulta. En una conversación, la API envía la primera pregunta del usuario (el tema).",
    )
    rerank: bool = Field(True, description="Reordenar los candidatos con el reranker BGE.")
    hierarchy: bool = Field(False, description="false = arquitectura B; true = C (agrega normas superiores).")
    document_ids: list[str] | None = Field(
        None, max_length=100, description="Limita la búsqueda base a estas normas (p. ej. [\"INT08\"]).",
    )
    turn: int | None = Field(None, ge=1, description="Solo para el registro: 1 = primera pregunta, >1 = seguimiento.")
    question_id: str | None = Field(None, max_length=100, description="Solo para el registro (Gold Standard).")
    tag: str | None = Field(None, max_length=100, description="Solo para el registro (experimento).")
    debug: bool = Field(False, description="Incluye tiempos por etapa, candidatos y detalle de la jerarquía.")

    @field_validator("question")
    @classmethod
    def _normalize_question(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("La pregunta está vacía.")
        return value

    @field_validator("document_ids")
    @classmethod
    def _unique_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = list(dict.fromkeys(item.strip().upper() for item in value if item.strip()))
        if not cleaned:
            raise ValueError("document_ids no puede estar vacío; omítalo para buscar en todo el corpus.")
        return cleaned


class UnitOut(BaseModel):
    unit_id: str
    document_id: str | None
    document_name: str | None
    rank: int | None
    label: str
    title: str
    text: str
    partial: bool
    score: float | None
    origin: Literal["base", "hierarchy"]
    matched_chunk_ids: list[str]


class Mode(BaseModel):
    rerank: bool
    hierarchy: bool
    architecture: Literal["B", "C"]


class RetrieveResponse(BaseModel):
    contract_version: str = CONTRACT_VERSION
    run_id: str
    baseline: str
    mode: Mode
    units: list[UnitOut]
    context: str
    tokens_estimated: int | None
    warnings: list[str]
    cached: bool
    index_version: dict[str, str]
    timings_ms: dict[str, Any]
    debug: dict[str, Any] | None = None

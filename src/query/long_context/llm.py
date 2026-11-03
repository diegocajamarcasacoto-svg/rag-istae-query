"""Contrato mínimo con el proveedor LLM, independiente de la marca.

Cada adaptador (un archivo por proveedor) traduce ``LLMRequest`` al formato de
su API respetando el orden recomendado para contextos largos:

    1. instrucciones del sistema (breves, describen la tarea)
    2. documentos completos                ← prefijo estable, reutilizable en caché
    3. historial de la conversación
    4. pregunta actual al final

Poner los documentos antes de la pregunta y la pregunta al final es la
recomendación de las guías de contexto largo y permite reutilizar la caché del
prefijo entre preguntas distintas.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMRequest:
    system: str
    context: str
    question: str
    history: tuple[ChatMessage, ...] = ()
    temperature: float = 0.0
    max_output_tokens: int = 2048
    cache_key: str | None = None  # identifica el prefijo (instrucciones + documentos)


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    extra: dict[str, object] = field(default_factory=dict)


class LLMClient(Protocol):
    """Interfaz que implementa cada adaptador de proveedor."""

    name: str
    model: str

    def count_tokens(self, request: LLMRequest) -> int:
        """Cuenta los tokens de entrada con el tokenizador oficial del proveedor."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Genera la respuesta completa."""

    def stream(self, request: LLMRequest) -> Iterator[str]:
        """Genera la respuesta por partes (para el chat)."""


def wrap_documents(context: str) -> str:
    """Delimita el corpus para que el modelo lo trate como datos."""
    return f"<documentos>\n{context}\n</documentos>"


def wrap_question(question: str) -> str:
    return f"<pregunta>\n{question.strip()}\n</pregunta>"

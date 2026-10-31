"""Contratos normalizados de prompts, respuestas y trazabilidad."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from query.context.models import ContextResult


@dataclass(frozen=True)
class GenerationPrompt:
    """Instrucción de sistema y contenido de usuario enviados al proveedor."""

    system_instruction: str
    user_content: str


@dataclass(frozen=True)
class ProviderResponse:
    """Respuesta normalizada del SDK, incluidos solo los tokens reportados."""

    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None


@dataclass(frozen=True)
class ProviderMetrics:
    """Métricas que el proveedor reporta; no incluye tiempos locales."""

    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class GenerationResult:
    """Respuesta junto con métricas y el resultado de contexto que la originó.

    ``context_result`` mantiene intactos los chunks y metadata de recuperación
    para que la respuesta pueda rastrearse sin enviar esa metadata al modelo.
    ``application_latency_ms`` se mide localmente y no se atribuye al proveedor.
    """

    answer: str
    provider: str
    model: str
    provider_metrics: ProviderMetrics
    application_latency_ms: float
    finish_reason: str | None
    context_result: ContextResult

    def to_dict(self) -> dict[str, Any]:
        """Serializa respuesta y trazabilidad completa del contexto de origen."""
        return {
            "answer": self.answer,
            "provider": self.provider,
            "model": self.model,
            "provider_metrics": {
                "input_tokens": self.provider_metrics.input_tokens,
                "output_tokens": self.provider_metrics.output_tokens,
            },
            "application_latency_ms": self.application_latency_ms,
            "finish_reason": self.finish_reason,
            "context_result": self.context_result.to_dict(),
        }

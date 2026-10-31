"""Normalización de respuestas del SDK Gemini a contratos del proyecto."""

from __future__ import annotations

from typing import Any

from query.generation.errors import GenerationProviderError
from query.generation.models import ProviderResponse


def _token_count(usage: Any, field: str) -> int | None:
    if usage is None:
        return None
    value = getattr(usage, field, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GenerationProviderError(
            f"Gemini devolvió un valor inválido para {field}: {value!r}."
        )
    return value


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None
    name = getattr(reason, "name", None)
    return name if isinstance(name, str) else str(reason)


def parse_provider_response(response: Any, *, fallback_model: str) -> ProviderResponse:
    """Extrae texto, versión de modelo, tokens y motivo del primer candidato."""
    finish_reason = _finish_reason(response)
    try:
        text = getattr(response, "text", None)
    except ValueError as exc:
        raise GenerationProviderError(
            f"Gemini no expuso texto en la respuesta (finish_reason={finish_reason})."
        ) from exc

    if not isinstance(text, str) or not text.strip():
        raise GenerationProviderError(
            f"Gemini devolvió una respuesta sin texto (finish_reason={finish_reason})."
        )

    usage = getattr(response, "usage_metadata", None)
    model = getattr(response, "model_version", None)
    if not isinstance(model, str) or not model.strip():
        model = fallback_model

    return ProviderResponse(
        text=text.strip(),
        model=model,
        input_tokens=_token_count(usage, "prompt_token_count"),
        output_tokens=_token_count(usage, "candidates_token_count"),
        finish_reason=finish_reason,
    )

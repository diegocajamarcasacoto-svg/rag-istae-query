"""Orquestación de una solicitud de generación sobre un ContextResult."""

from __future__ import annotations

import time

from query.context.models import ContextResult
from query.generation.errors import GenerationConfigurationError
from query.generation.generation_config import GenerationConfig
from query.generation.models import GenerationResult, ProviderMetrics
from query.generation.prompt_builder import build_prompt, load_system_prompt
from query.generation.providers.base import GenerationProvider
from query.generation.providers.gemini import GeminiProvider


def generate_answer(
    context_result: ContextResult,
    config: GenerationConfig,
    *,
    provider: GenerationProvider | None = None,
) -> GenerationResult:
    """Genera una respuesta y conserva el ContextResult que la fundamenta.

    Solo la consulta y el texto consolidado se mandan al proveedor. Los chunks,
    scores y metadata permanecen adjuntos a ``GenerationResult`` para auditoría.
    El tiempo medido cubre la solicitud al proveedor y se reporta aparte de los
    tokens de entrada/salida que el propio proveedor informe.
    """
    if not isinstance(context_result, ContextResult):
        raise TypeError("context_result debe ser una instancia de ContextResult.")
    if not isinstance(config, GenerationConfig):
        raise TypeError("config debe ser una instancia de GenerationConfig.")

    selected_provider = provider
    if selected_provider is None:
        if config.provider != "gemini":
            raise GenerationConfigurationError(
                f"Proveedor no implementado: {config.provider!r}."
            )
        selected_provider = GeminiProvider()

    system_instruction = load_system_prompt(config.system_prompt_path)
    prompt = build_prompt(context_result, system_instruction)
    started_at = time.perf_counter()
    response = selected_provider.generate(prompt, config=config)
    application_latency_ms = (time.perf_counter() - started_at) * 1000

    return GenerationResult(
        answer=response.text,
        provider=selected_provider.name,
        model=response.model,
        provider_metrics=ProviderMetrics(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        ),
        application_latency_ms=application_latency_ms,
        finish_reason=response.finish_reason,
        context_result=context_result,
    )

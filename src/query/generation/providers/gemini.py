"""Adaptador síncrono del SDK oficial google-genai para Gemini."""

from __future__ import annotations

import math
import os
from typing import Any

from query.generation.errors import GenerationConfigurationError
from query.generation.generation_config import GenerationConfig
from query.generation.models import GenerationPrompt, ProviderResponse
from query.generation.response_parser import parse_provider_response


class GeminiProvider:
    """Invoca Gemini mediante ``models.generate_content``.

    El SDK y el cliente HTTP se importan/crean únicamente cuando se necesita
    generar una respuesta. Inyectar ``client`` permite probar el adaptador
    sin instalar el SDK ni realizar solicitudes de red.
    """

    name = "gemini"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def generate(
        self,
        prompt: GenerationPrompt,
        *,
        config: GenerationConfig,
    ) -> ProviderResponse:
        """Envía pregunta y contexto, con System Prompt y límites explícitos."""
        client = self._client if self._client is not None else self._create_client(config)
        response = client.models.generate_content(
            model=config.model,
            contents=prompt.user_content,
            config={
                "system_instruction": prompt.system_instruction,
                "temperature": config.temperature,
                "max_output_tokens": config.max_output_tokens,
            },
        )
        return parse_provider_response(response, fallback_model=config.model)

    @staticmethod
    def _create_client(config: GenerationConfig) -> Any:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise GenerationConfigurationError(
                f"No está definida la variable de entorno {config.api_key_env}."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GenerationConfigurationError(
                "Falta google-genai; instale las dependencias de requirements.txt."
            ) from exc

        timeout_ms = math.ceil(config.timeout_seconds * 1000)
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

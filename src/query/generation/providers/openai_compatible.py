"""Adaptador HTTP para APIs compatibles con OpenAI Chat Completions.

Se usa para OpenRouter (p. ej. ``nvidia/nemotron-3-ultra-550b-a55b:free``),
pero el formato es el mismo que exponen OpenAI, vLLM u Ollama, por lo que basta
con cambiar ``base_url``, ``model`` y ``api_key_env`` en la configuración.
"""

from __future__ import annotations

import os
from typing import Any

from query.generation.errors import (
    GenerationConfigurationError,
    GenerationProviderError,
)
from query.generation.generation_config import GenerationConfig
from query.generation.models import GenerationPrompt, ProviderResponse


def _token_count(usage: Any, field: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GenerationProviderError(
            f"El proveedor devolvió un valor inválido para {field}: {value!r}."
        )
    return value


def _error_message(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message") or error
        code = error.get("code")
        return f"{message} (código {code})" if code is not None else str(message)
    return str(error)


def parse_chat_completion(data: Any, *, fallback_model: str) -> ProviderResponse:
    """Normaliza un JSON de Chat Completions a ``ProviderResponse``."""
    if not isinstance(data, dict):
        raise GenerationProviderError("El proveedor devolvió un JSON que no es un objeto.")
    if data.get("error"):
        raise GenerationProviderError(
            f"El proveedor devolvió un error: {_error_message(data['error'])}"
        )

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise GenerationProviderError("El proveedor devolvió una respuesta sin choices.")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    if choice.get("error"):
        raise GenerationProviderError(
            f"El proveedor devolvió un error: {_error_message(choice['error'])}"
        )

    message = choice.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise GenerationProviderError(
            f"El proveedor devolvió una respuesta sin texto (finish_reason={finish_reason})."
        )

    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        model = fallback_model

    usage = data.get("usage")
    return ProviderResponse(
        text=text.strip(),
        model=model,
        input_tokens=_token_count(usage, "prompt_tokens"),
        output_tokens=_token_count(usage, "completion_tokens"),
        finish_reason=finish_reason,
    )


class OpenAICompatibleProvider:
    """Invoca ``POST {base_url}/chat/completions`` con httpx.

    El cliente HTTP se crea de forma diferida. Inyectar ``client`` (cualquier
    objeto con ``post(url, *, json, headers, timeout)`` que devuelva una
    respuesta con ``status_code`` y ``json()``) permite probar sin red.
    """

    def __init__(self, client: Any | None = None, *, name: str = "openrouter") -> None:
        self._client = client
        self.name = name

    def generate(
        self,
        prompt: GenerationPrompt,
        *,
        config: GenerationConfig,
    ) -> ProviderResponse:
        """Envía System Prompt y contenido de usuario como mensajes de chat."""
        if not config.base_url:
            raise GenerationConfigurationError(
                f"El proveedor {config.provider!r} requiere base_url en la configuración."
            )
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise GenerationConfigurationError(
                f"No está definida la variable de entorno {config.api_key_env}."
            )

        body: dict[str, Any] = {
            **config.extra_body,
            "model": config.model,
            "messages": [
                {"role": "system", "content": prompt.system_instruction},
                {"role": "user", "content": prompt.user_content},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
        }
        headers = {
            **config.extra_headers,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{config.base_url}/chat/completions"

        client = self._client if self._client is not None else self._create_client()
        try:
            response = client.post(
                url, json=body, headers=headers, timeout=config.timeout_seconds
            )
        except Exception as exc:  # errores de red/timeout del cliente HTTP
            raise GenerationProviderError(
                f"No se pudo contactar al proveedor {self.name!r}: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError:
            data = None

        status = getattr(response, "status_code", 200)
        if status >= 400:
            detail = (
                _error_message(data.get("error"))
                if isinstance(data, dict) and data.get("error")
                else getattr(response, "text", "")
            )
            hint = " Límite de tasa alcanzado; reintente más tarde." if status == 429 else ""
            raise GenerationProviderError(
                f"El proveedor {self.name!r} respondió HTTP {status}: {detail}.{hint}"
            )
        if data is None:
            raise GenerationProviderError("El proveedor devolvió una respuesta que no es JSON.")
        return parse_chat_completion(data, fallback_model=config.model)

    @staticmethod
    def _create_client() -> Any:
        try:
            import httpx
        except ImportError as exc:
            raise GenerationConfigurationError(
                "Falta httpx; instale las dependencias de requirements.txt."
            ) from exc
        return httpx.Client()

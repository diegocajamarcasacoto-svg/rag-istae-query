"""Adaptador de Gemini (SDK oficial ``google-genai``) para ``LLMClient``.

Orden del prompt (recomendado para contextos largos):
    instrucción del sistema = instrucciones + <documentos>…</documentos>
    contenidos              = historial de la conversación + <pregunta>…</pregunta>

Así el prefijo estable (instrucciones + corpus) queda en la instrucción del
sistema y puede guardarse en la caché de contexto de Gemini, mientras que lo que
cambia en cada consulta (historial y pregunta) va al final.

La caché explícita es opcional (``use_cache``): según el plan de la cuenta
puede no estar disponible. Se crea una vez por ``cache_key`` (la versión del
corpus) y se reutiliza mientras dure su TTL. Verificar en la primera ejecución
real que la creación de la caché funciona con la cuenta usada.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

from query.long_context.llm import LLMRequest, LLMResponse, wrap_documents, wrap_question


class GeminiConfigurationError(RuntimeError):
    """Falta una credencial, dependencia o parámetro para usar Gemini."""


class GeminiClient:
    name = "gemini"

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "GEMINI_API_KEY",
        use_cache: bool = False,
        cache_ttl_seconds: int = 600,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        send_temperature: bool = True,
        timeout_seconds: float = 300,
        client: Any | None = None,
    ) -> None:
        if thinking_budget is not None and thinking_level is not None:
            raise GeminiConfigurationError("Use thinking_level o thinking_budget, no ambos (la API rechaza la combinación).")
        # Gemini 3: Google recomienda no cambiar la temperatura (1.0 por defecto);
        # con send_temperature=False se usa el valor por defecto del modelo.
        self.thinking_level = thinking_level
        self.send_temperature = send_temperature
        if not model or not model.strip():
            raise GeminiConfigurationError("Debe indicar el nombre exacto del modelo de Gemini.")
        self.model = model.strip()
        self.api_key_env = api_key_env
        self.use_cache = use_cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self.thinking_budget = thinking_budget
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._caches: dict[str, str] = {}

    # --- infraestructura -------------------------------------------------
    @staticmethod
    def _types() -> Any:
        try:
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise GeminiConfigurationError("Instale el SDK: pip install google-genai") from exc
        return types

    def _sdk(self) -> Any:
        if self._client is None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise GeminiConfigurationError(f"No está definida la variable de entorno {self.api_key_env}.")
            from google import genai

            types = self._types()
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
            )
        return self._client

    # --- armado del prompt -----------------------------------------------
    @staticmethod
    def system_text(request: LLMRequest) -> str:
        return f"{request.system}\n\n{wrap_documents(request.context)}"

    def contents(self, request: LLMRequest) -> list[Any]:
        types = self._types()
        items = [
            types.Content(role="model" if message.role == "assistant" else "user", parts=[types.Part(text=message.content)])
            for message in request.history
        ]
        items.append(types.Content(role="user", parts=[types.Part(text=wrap_question(request.question, request.reminder))]))
        return items

    def _thinking(self) -> Any | None:
        if self.thinking_level is not None:
            return self._types().ThinkingConfig(thinking_level=self.thinking_level)
        if self.thinking_budget is None:
            return None
        return self._types().ThinkingConfig(thinking_budget=self.thinking_budget)

    def _ensure_cache(self, request: LLMRequest) -> str:
        key = request.cache_key or "sin-clave"
        if key not in self._caches:
            types = self._types()
            cache = self._sdk().caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=self.system_text(request),
                    ttl=f"{int(self.cache_ttl_seconds)}s",
                    display_name=f"istae-{key}",
                ),
            )
            self._caches[key] = cache.name
        return self._caches[key]

    def generation_config(self, request: LLMRequest) -> Any:
        types = self._types()
        options: dict[str, Any] = {"max_output_tokens": request.max_output_tokens}
        if self.send_temperature:
            options["temperature"] = request.temperature
        thinking = self._thinking()
        if thinking is not None:
            options["thinking_config"] = thinking
        if self.use_cache:
            options["cached_content"] = self._ensure_cache(request)
        else:
            options["system_instruction"] = self.system_text(request)
        return types.GenerateContentConfig(**options)

    # --- interfaz LLMClient ----------------------------------------------
    def count_tokens(self, request: LLMRequest) -> int:
        """Cuenta los tokens de entrada con el tokenizador oficial del modelo.

        Las instrucciones y el corpus se cuentan como un contenido más (el
        resultado difiere en pocos tokens del conteo como instrucción del sistema).
        """
        types = self._types()
        prefix = types.Content(role="user", parts=[types.Part(text=self.system_text(request))])
        response = self._sdk().models.count_tokens(model=self.model, contents=[prefix, *self.contents(request)])
        return int(response.total_tokens)

    def context_window(self) -> int | None:
        """Límite de tokens de entrada que informa la API para el modelo."""
        model = self._sdk().models.get(model=self.model)
        limit = getattr(model, "input_token_limit", None)
        return int(limit) if limit else None

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        response = self._sdk().models.generate_content(
            model=self.model, contents=self.contents(request), config=self.generation_config(request)
        )
        return self._to_response(response, (time.perf_counter() - started) * 1000)

    def stream(self, request: LLMRequest) -> Iterator[str]:
        for chunk in self._sdk().models.generate_content_stream(
            model=self.model, contents=self.contents(request), config=self.generation_config(request)
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text

    def close(self) -> None:
        """Elimina las cachés creadas por este cliente (evita cobro de almacenamiento)."""
        if self._client is None:
            return
        for name in list(self._caches.values()):
            try:
                self._client.caches.delete(name=name)
            finally:
                self._caches = {k: v for k, v in self._caches.items() if v != name}

    # --- utilidades ------------------------------------------------------
    def _to_response(self, response: Any, latency_ms: float) -> LLMResponse:
        usage = getattr(response, "usage_metadata", None)
        candidates = getattr(response, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        reason_name = getattr(reason, "name", None) or (str(reason) if reason is not None else None)
        thoughts = getattr(usage, "thoughts_token_count", None) if usage else None
        served = getattr(response, "model_version", None) or self.model
        # Google redirige modelos deprecados a otros sin error: se registra si el
        # modelo que respondió no es el configurado, para no mezclar resultados.
        mismatch = not str(served).startswith(self.model)
        return LLMResponse(
            text=(getattr(response, "text", None) or "").strip(),
            model=served,
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            cached_input_tokens=getattr(usage, "cached_content_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            finish_reason=reason_name,
            latency_ms=latency_ms,
            extra={
                "thoughts_tokens": thoughts,
                "cache_used": self.use_cache,
                "requested_model": self.model,
                "model_mismatch": mismatch,
            },
        )

"""Adaptador para APIs compatibles con OpenAI (OpenRouter, NVIDIA NIM, DeepSeek…).

Usa el endpoint estándar ``POST {base_url}/chat/completions`` con ``httpx``
(ya instalado como dependencia de google-genai), sin SDK adicional.

Orden del prompt, igual que en el adaptador de Gemini:
    mensaje de sistema = instrucciones + <documentos>…</documentos>
    mensajes           = historial + <pregunta>…</pregunta>

Estos proveedores no exponen un contador de tokens oficial: ``count_tokens``
devuelve una estimación (caracteres / 4,44, la razón medida con el tokenizador
de Gemini sobre este corpus). El conteo real llega en ``usage`` de cada respuesta.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

from query.long_context.llm import LLMRequest, LLMResponse, wrap_documents, wrap_question

CHARS_PER_TOKEN = 4.44
DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "deepseek": "https://api.deepseek.com",
}


class OpenAICompatibleError(RuntimeError):
    """Error de configuración o respuesta del proveedor."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = status_code  # mismo atributo que usan los errores de google-genai


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str,
        *,
        provider: str = "openrouter",
        base_url: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        send_temperature: bool = False,
        reasoning_effort: str | None = None,
        timeout_seconds: float = 600,
        http_client: Any | None = None,
    ) -> None:
        if not model or not model.strip():
            raise OpenAICompatibleError("Debe indicar el nombre exacto del modelo.")
        self.name = provider
        self.model = model.strip()
        self.base_url = (base_url or DEFAULT_BASE_URLS.get(provider) or "").rstrip("/")
        if not self.base_url:
            raise OpenAICompatibleError(f"Indique base_url para el proveedor {provider!r}.")
        self.api_key_env = api_key_env
        self.send_temperature = send_temperature
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self._http = http_client

    # --- infraestructura -------------------------------------------------
    def _client(self) -> Any:
        if self._http is None:
            import httpx

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise OpenAICompatibleError(f"No está definida la variable de entorno {self.api_key_env}.")
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            if self.name == "openrouter":
                headers["X-Title"] = "Tesis ISTAE - contexto largo"
            self._http = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout_seconds, connect=30),
            )
        return self._http

    # --- armado del prompt -----------------------------------------------
    @staticmethod
    def system_text(request: LLMRequest) -> str:
        return f"{request.system}\n\n{wrap_documents(request.context)}"

    def messages(self, request: LLMRequest) -> list[dict[str, str]]:
        items = [{"role": "system", "content": self.system_text(request)}]
        items += [{"role": m.role, "content": m.content} for m in request.history]
        items.append({"role": "user", "content": wrap_question(request.question, request.reminder)})
        return items

    def payload(self, request: LLMRequest, stream: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages(request),
            "max_tokens": request.max_output_tokens,
        }
        if self.send_temperature:
            body["temperature"] = request.temperature
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if self.name == "openrouter":
            body["usage"] = {"include": True}
        if stream:
            body["stream"] = True
        return body

    def _raise_for_error(self, response: Any) -> None:
        if response.status_code < 400:
            return
        try:
            detail = response.json().get("error", {})
            message = detail.get("message") or response.text
            raw = (detail.get("metadata") or {}).get("raw")
            if raw:
                message = f"{message} | {raw}"
        except ValueError:
            message = response.text
        raise OpenAICompatibleError(f"{response.status_code} {message}", response.status_code)

    # --- interfaz LLMClient ----------------------------------------------
    def count_tokens(self, request: LLMRequest) -> int:
        """Estimación (sin tokenizador oficial disponible por API)."""
        chars = sum(len(m["content"]) for m in self.messages(request))
        return round(chars / CHARS_PER_TOKEN)

    def context_window(self) -> int | None:
        if self.name != "openrouter":
            return None
        response = self._client().get("/models")
        self._raise_for_error(response)
        for item in response.json().get("data", []):
            if item.get("id") == self.model:
                return item.get("context_length")
        return None

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        response = self._client().post("/chat/completions", json=self.payload(request))
        self._raise_for_error(response)
        data = response.json()
        if "error" in data:  # algunos proveedores devuelven 200 con error en el cuerpo
            raise OpenAICompatibleError(str(data["error"]), (data["error"] or {}).get("code"))
        return self._to_response(data, (time.perf_counter() - started) * 1000)

    def stream(self, request: LLMRequest) -> Iterator[str]:
        with self._client().stream("POST", "/chat/completions", json=self.payload(request, stream=True)) as response:
            if response.status_code >= 400:
                response.read()
                self._raise_for_error(response)
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue  # comentarios de keep-alive de OpenRouter
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                event = json.loads(chunk)
                if "error" in event:
                    raise OpenAICompatibleError(str(event["error"]))
                for choice in event.get("choices", []):
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        yield text

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    # --- utilidades ------------------------------------------------------
    def _to_response(self, data: dict[str, Any], latency_ms: float) -> LLMResponse:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        served = data.get("model") or self.model
        return LLMResponse(
            text=(message.get("content") or "").strip(),
            model=served,
            input_tokens=usage.get("prompt_tokens"),
            cached_input_tokens=prompt_details.get("cached_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            latency_ms=latency_ms,
            extra={
                "thoughts_tokens": completion_details.get("reasoning_tokens"),
                "provider": data.get("provider"),
                "cost_usd": usage.get("cost"),
                "requested_model": self.model,
                "model_mismatch": served.removesuffix(":free") != self.model.removesuffix(":free"),
            },
        )

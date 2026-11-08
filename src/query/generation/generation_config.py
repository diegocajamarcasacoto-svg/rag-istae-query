"""Carga y valida la configuración independiente del módulo de generación."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Valores predeterminados por proveedor. Un proveedor nuevo solo necesita una
# entrada aquí (si requiere defaults) y otra en ``query.generation.providers``.
DEFAULT_API_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
DEFAULT_BASE_URL = {
    "openrouter": "https://openrouter.ai/api/v1",
}
# modular: mismo armado que rag-istae-api (system_prompt_path es una carpeta con
#   base_es.txt, rag_es.txt, jerarquia_es.txt y recordatorio_es.txt). Es la única
#   estrategia: la anterior "single" (un System Prompt y el contexto como JSON) se
#   retiró para que B/C se evalúen siempre con el prompt de la API.
PROMPT_STRATEGIES = ("modular",)


@dataclass(frozen=True)
class GenerationConfig:
    """Opciones para una solicitud de generación.

    ``system_prompt_path`` es la carpeta de prompts modulares y se resuelve
    respecto del archivo JSON que declara la ruta. La credencial no se almacena aquí: ``api_key_env`` nombra la
    variable de entorno que consulta el adaptador del proveedor.

    ``base_url``, ``extra_headers`` y ``extra_body`` solo los usan los
    proveedores HTTP compatibles con OpenAI (p. ej. OpenRouter); Gemini los
    ignora. ``extra_body`` se mezcla en el cuerpo de la solicitud.
    """

    provider: str
    model: str
    system_prompt_path: Path
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    api_key_env: str
    base_url: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    prompt_strategy: str = "modular"


def load_generation_config(path: Path) -> GenerationConfig:
    """Carga la configuración JSON sin acoplarla a ``query.config``.

    Los parámetros numéricos se validan antes de crear cualquier cliente.
    Las rutas relativas del prompt se anclan al directorio del JSON, no al
    directorio de trabajo del proceso.
    """
    config_path = Path(path).expanduser().resolve()
    data: Any = json.loads(config_path.read_text(encoding="utf-8-sig"))  # con o sin BOM
    if not isinstance(data, dict):
        raise TypeError("La configuración de generación debe ser un objeto JSON.")

    required = ("provider", "model", "system_prompt_path")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise KeyError(f"Faltan claves de configuración: {', '.join(missing)}")

    provider = data["provider"]
    model = data["model"]
    prompt_value = data["system_prompt_path"]
    if not isinstance(provider, str) or not provider.strip():
        raise TypeError("provider debe ser un texto no vacío.")
    if not isinstance(model, str) or not model.strip():
        raise TypeError("model debe ser un texto no vacío.")
    if not isinstance(prompt_value, str) or not prompt_value.strip():
        raise TypeError("system_prompt_path debe ser una ruta no vacía.")

    temperature = data.get("temperature", 0.2)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("temperature debe ser un número finito entre 0 y 2.")

    max_output_tokens = data.get("max_output_tokens", 2048)
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens <= 0
    ):
        raise ValueError("max_output_tokens debe ser un entero positivo.")

    timeout_seconds = data.get("timeout_seconds", 60)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds debe ser un número finito positivo.")

    provider_name = provider.strip().lower()
    api_key_env = data.get(
        "api_key_env",
        DEFAULT_API_KEY_ENV.get(provider_name, f"{provider_name.upper()}_API_KEY"),
    )
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise TypeError("api_key_env debe ser un nombre de variable no vacío.")

    base_url = data.get("base_url", DEFAULT_BASE_URL.get(provider_name))
    if base_url is not None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise TypeError("base_url debe ser una URL no vacía.")
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("base_url debe comenzar con http:// o https://.")

    extra_headers = data.get("extra_headers", {})
    if not isinstance(extra_headers, dict) or not all(
        isinstance(key, str) and key.strip() and isinstance(value, str)
        for key, value in extra_headers.items()
    ):
        raise TypeError("extra_headers debe ser un objeto JSON de textos.")

    extra_body = data.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise TypeError("extra_body debe ser un objeto JSON.")

    prompt_strategy = data.get("prompt_strategy", "modular")
    if prompt_strategy not in PROMPT_STRATEGIES:
        raise ValueError(
            f"prompt_strategy debe ser uno de: {', '.join(PROMPT_STRATEGIES)}. La estrategia "
            "'single' se retiró: use system_prompt_path = \"prompts/modular\"."
        )

    prompt_path = Path(prompt_value).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = config_path.parent / prompt_path
    if prompt_path.is_file():
        raise ValueError(
            f"system_prompt_path debe ser la carpeta de prompts modulares, no un archivo: {prompt_path}"
        )

    return GenerationConfig(
        provider=provider_name,
        model=model.strip(),
        system_prompt_path=prompt_path.resolve(),
        temperature=float(temperature),
        max_output_tokens=max_output_tokens,
        timeout_seconds=float(timeout_seconds),
        api_key_env=api_key_env.strip(),
        base_url=base_url,
        extra_headers=dict(extra_headers),
        extra_body=dict(extra_body),
        prompt_strategy=prompt_strategy,
    )

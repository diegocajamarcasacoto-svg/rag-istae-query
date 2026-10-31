"""Carga y valida la configuración independiente del módulo de generación."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    """Opciones para una solicitud de generación.

    ``system_prompt_path`` se resuelve respecto del archivo JSON que declara
    la ruta. La credencial no se almacena aquí: ``api_key_env`` nombra la
    variable de entorno que consulta el adaptador del proveedor.
    """

    provider: str
    model: str
    system_prompt_path: Path
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    api_key_env: str


def load_generation_config(path: Path) -> GenerationConfig:
    """Carga la configuración JSON sin acoplarla a ``query.config``.

    Los parámetros numéricos se validan antes de crear cualquier cliente.
    Las rutas relativas del prompt se anclan al directorio del JSON, no al
    directorio de trabajo del proceso.
    """
    config_path = Path(path).expanduser().resolve()
    data: Any = json.loads(config_path.read_text(encoding="utf-8"))
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

    api_key_env = data.get("api_key_env", "GEMINI_API_KEY")
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise TypeError("api_key_env debe ser un nombre de variable no vacío.")

    prompt_path = Path(prompt_value).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = config_path.parent / prompt_path

    return GenerationConfig(
        provider=provider.strip().lower(),
        model=model.strip(),
        system_prompt_path=prompt_path.resolve(),
        temperature=float(temperature),
        max_output_tokens=max_output_tokens,
        timeout_seconds=float(timeout_seconds),
        api_key_env=api_key_env.strip(),
    )

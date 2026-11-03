"""Configuración de la variante de contexto largo (archivo JSON, sin credenciales).

Las rutas relativas se resuelven respecto del propio archivo de configuración.
La API key nunca se guarda aquí: se lee de la variable indicada en ``api_key_env``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from query.long_context.llm import LLMClient


@dataclass(frozen=True)
class LongContextSettings:
    provider: str
    model: str
    api_key_env: str
    temperature: float | None
    max_output_tokens: int
    use_cache: bool
    cache_ttl_seconds: int
    thinking_budget: int | None
    thinking_level: str | None
    timeout_seconds: float
    max_history_messages: int
    base_url: str | None
    prompts_dir: Path
    catalog: Path
    outputs_dir: Path
    results_dir: Path


def _path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_settings(path: Path) -> LongContextSettings:
    path = Path(path).expanduser().resolve()
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))  # acepta BOM (Bloc de notas)
    missing = [key for key in ("provider", "model", "outputs_dir") if not data.get(key)]
    if missing:
        raise KeyError(f"Faltan claves en {path.name}: {', '.join(missing)}")
    base = path.parent
    budget = data.get("thinking_budget")
    outputs_dir = _path(base, data["outputs_dir"])
    # Por defecto los resultados quedan junto a los corpus: <outputs>/long_context/resultados
    results_dir = _path(base, data["results_dir"]) if data.get("results_dir") else outputs_dir / "long_context" / "resultados"
    return LongContextSettings(
        provider=str(data["provider"]).strip().lower(),
        model=str(data["model"]).strip(),
        api_key_env=str(data.get("api_key_env", "GEMINI_API_KEY")),
        # null = valor por defecto del modelo (recomendado por Google para Gemini 3).
        temperature=None if data.get("temperature") is None else float(data["temperature"]),
        max_output_tokens=int(data.get("max_output_tokens", 2048)),
        use_cache=bool(data.get("use_cache", False)),
        cache_ttl_seconds=int(data.get("cache_ttl_seconds", 600)),
        thinking_budget=None if budget is None else int(budget),
        thinking_level=data.get("thinking_level") or None,
        timeout_seconds=float(data.get("timeout_seconds", 300)),
        max_history_messages=int(data.get("max_history_messages", 6)),
        base_url=data.get("base_url") or None,
        prompts_dir=_path(base, data.get("prompts_dir", "prompts")),
        catalog=_path(base, data.get("catalog", "catalogo_normativo.json")),
        outputs_dir=outputs_dir,
        results_dir=results_dir,
    )


def build_client(settings: LongContextSettings) -> LLMClient:
    if settings.provider == "gemini":
        from query.long_context.providers.gemini import GeminiClient

        return GeminiClient(
            settings.model,
            api_key_env=settings.api_key_env,
            use_cache=settings.use_cache,
            cache_ttl_seconds=settings.cache_ttl_seconds,
            thinking_budget=settings.thinking_budget,
            thinking_level=settings.thinking_level,
            send_temperature=settings.temperature is not None,
            timeout_seconds=settings.timeout_seconds,
        )
    if settings.provider in ("openrouter", "nvidia", "deepseek", "openai_compatible"):
        from query.long_context.providers.openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient(
            settings.model,
            provider=settings.provider,
            base_url=settings.base_url,
            api_key_env=settings.api_key_env,
            send_temperature=settings.temperature is not None,
            reasoning_effort=settings.thinking_level,
            timeout_seconds=settings.timeout_seconds,
        )
    raise ValueError(f"Proveedor no implementado: {settings.provider!r}")

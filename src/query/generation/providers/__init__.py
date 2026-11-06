"""Adaptadores de proveedores del módulo de generación.

Para agregar un proveedor: implementar ``GenerationProvider`` (``base.py``) y
registrarlo en ``PROVIDERS``. La elección se hace con la clave ``provider`` del
JSON de generación; no se requiere modificar el resto del pipeline.
"""

from __future__ import annotations

from typing import Callable

from query.generation.errors import GenerationConfigurationError
from query.generation.providers.base import GenerationProvider
from query.generation.providers.gemini import GeminiProvider
from query.generation.providers.openai_compatible import OpenAICompatibleProvider

PROVIDERS: dict[str, Callable[[], GenerationProvider]] = {
    "gemini": GeminiProvider,
    "openrouter": lambda: OpenAICompatibleProvider(name="openrouter"),
}


def create_provider(name: str) -> GenerationProvider:
    """Instancia el adaptador registrado con ``name``."""
    factory = PROVIDERS.get(name)
    if factory is None:
        available = ", ".join(sorted(PROVIDERS))
        raise GenerationConfigurationError(
            f"Proveedor no implementado: {name!r}. Disponibles: {available}."
        )
    return factory()


__all__ = [
    "PROVIDERS",
    "GeminiProvider",
    "GenerationProvider",
    "OpenAICompatibleProvider",
    "create_provider",
]

"""Protocolo común para adaptadores de generación."""

from __future__ import annotations

from typing import Protocol

from query.generation.generation_config import GenerationConfig
from query.generation.models import GenerationPrompt, ProviderResponse


class GenerationProvider(Protocol):
    """Interfaz mínima que debe implementar un proveedor de generación."""

    name: str

    def generate(
        self,
        prompt: GenerationPrompt,
        *,
        config: GenerationConfig,
    ) -> ProviderResponse:
        """Ejecuta una solicitud y devuelve una respuesta normalizada."""
        ...

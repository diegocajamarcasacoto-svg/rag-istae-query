"""Construcción del contenido enviado al modelo y carga de prompt externo."""

from __future__ import annotations

import json
from pathlib import Path

from query.context.models import ContextResult
from query.generation.models import GenerationPrompt


def load_system_prompt(path: Path) -> str:
    """Lee el System Prompt externo y rechaza archivos vacíos o ausentes."""
    prompt = Path(path).read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"El System Prompt está vacío: {path}")
    return prompt.strip()


def build_prompt(
    context_result: ContextResult,
    system_instruction: str,
) -> GenerationPrompt:
    """Empaqueta consulta y texto de contexto como datos JSON no confiables.

    Solo se serializan ``ContextResult.query`` y ``ContextResult.context``.
    Los chunks y metadata se conservan en el resultado final, pero no se
    incorporan al contenido enviado al proveedor.
    """
    if not isinstance(context_result, ContextResult):
        raise TypeError("context_result debe ser una instancia de ContextResult.")
    if not isinstance(context_result.query, str) or not context_result.query.strip():
        raise ValueError("La consulta del contexto debe ser un texto no vacío.")
    if not isinstance(context_result.context, str):
        raise TypeError("context_result.context debe ser texto.")
    if not isinstance(system_instruction, str) or not system_instruction.strip():
        raise ValueError("system_instruction debe ser un texto no vacío.")

    user_content = json.dumps(
        {
            "question": context_result.query,
            "reference_context": context_result.context,
        },
        ensure_ascii=False,
        indent=2,
    )
    return GenerationPrompt(
        system_instruction=system_instruction.strip(),
        user_content=user_content,
    )

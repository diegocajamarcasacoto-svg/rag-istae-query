"""Construcción del contenido enviado al modelo con los prompts modulares (formato de rag-istae-api)."""

from __future__ import annotations

from pathlib import Path

from query.context.models import ContextResult
from query.generation.models import GenerationPrompt

MODULAR_BASE = "base_es.txt"
MODULAR_RAG = "rag_es.txt"
MODULAR_HIERARCHY = "jerarquia_es.txt"
MODULAR_REMINDER = "recordatorio_es.txt"


def load_system_prompt(path: Path) -> str:
    """Lee una parte del prompt y rechaza archivos vacíos o ausentes."""
    prompt = Path(path).read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"El prompt está vacío: {path}")
    return prompt.strip()


def load_modular_prompts(prompts_dir: Path, hierarchical: bool) -> tuple[str, str]:
    """Instrucciones y recordatorio con el mismo armado que ``rag-istae-api``.

    B (``hybrid``): base + rag. C (``hierarchical``): base + rag + jerarquía.
    El recordatorio va al final del mensaje del usuario.
    """
    prompts_dir = Path(prompts_dir)
    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de prompts: {prompts_dir}")
    names = [MODULAR_BASE, MODULAR_RAG] + ([MODULAR_HIERARCHY] if hierarchical else [])
    system = "\n\n".join(load_system_prompt(prompts_dir / name) for name in names)
    return system, load_system_prompt(prompts_dir / MODULAR_REMINDER)


def build_modular_prompt(context_result: ContextResult, prompts_dir: Path) -> GenerationPrompt:
    """Mensajes como en ``rag-istae-api``: instrucciones + ``<documentos>`` en el
    sistema; ``<pregunta>`` + recordatorio en el usuario.

    Requiere el contexto presentado por documentos (``query.presentation``), que
    es lo que hace válidas las reglas de cita del prompt base.
    """
    if not isinstance(context_result, ContextResult):
        raise TypeError("context_result debe ser una instancia de ContextResult.")
    if context_result.metadata.get("presentation") != "documents":
        raise ValueError(
            "prompt_strategy=modular requiere el contexto por documentos (--expansion units)."
        )
    question = context_result.query.strip() if isinstance(context_result.query, str) else ""
    if not question:
        raise ValueError("La consulta del contexto debe ser un texto no vacío.")
    hierarchical = context_result.metadata.get("retrieval_mode") == "hierarchical"
    system, reminder = load_modular_prompts(prompts_dir, hierarchical)
    return GenerationPrompt(
        system_instruction=f"{system}\n\n<documentos>\n{context_result.context}\n</documentos>",
        user_content=f"<pregunta>\n{question}\n</pregunta>\n\n{reminder}",
    )

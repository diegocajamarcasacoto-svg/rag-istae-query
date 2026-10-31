"""Presupuesto configurable por chunk completo y estado de conteo explícito."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from query.context.context_merger import chunk_identity
from query.context.models import ContextChunk


def _token_counter(config: dict[str, Any]) -> Callable[[str], int] | None:
    counter = config.get("token_counter")
    if callable(counter):
        return lambda text: _count_callable(counter, text)
    tokenizer = config.get("tokenizer")
    if tokenizer is not None:
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            return lambda text: len(encode(text))
        if callable(tokenizer):
            return lambda text: _count_callable(tokenizer, text)
        raise TypeError("tokenizer debe ser callable o exponer encode(text).")
    return None


def _count_callable(counter: Callable[[str], Any], text: str) -> int:
    result = counter(text)
    if isinstance(result, int) and not isinstance(result, bool):
        count = result
    else:
        try:
            count = len(result)
        except TypeError as exc:
            raise TypeError("El tokenizador debe devolver un entero o una secuencia.") from exc
    if count < 0:
        raise ValueError("El conteo de tokens no puede ser negativo.")
    return count


def _estimated_token_count(text: str) -> int:
    return math.ceil(len(text.encode("utf-8")) / 4) if text else 0


def apply_token_budget(
    chunks: list[ContextChunk],
    config: dict[str, Any],
) -> tuple[list[ContextChunk], list[ContextChunk], dict[str, Any]]:
    """Aplica ``max_tokens`` sin truncar chunks documentales.

    Puede configurarse ``config["tokenizer"]`` con método ``encode(text)`` o
    como callable. Alternativamente, ``config["token_counter"]`` puede
    devolver un entero o una secuencia con longitud. Si no se configura ninguno,
    se estiman tokens como bytes UTF-8 del contexto divididos por cuatro,
    redondeados hacia arriba, y se marca el resultado como estimado.

    Todas las semillas se retienen antes de aplicar el presupuesto, incluso si
    por sí solas lo exceden. Los demás chunks se evalúan en el orden documental
    recibido y se agregan solo si el conteo del contexto unido —con separadores
    de doble salto de línea— queda dentro del límite. Se vuelve a contar el
    texto completo para contemplar tokenizadores cuyo resultado dependa de la
    frontera entre chunks. Los excluidos se devuelven completos para registrar
    el motivo desde la capa que invoca esta función.

    Args:
        chunks: Chunks deduplicados y ordenados para ensamblar el contexto.
        config: Opciones del builder con ``max_tokens`` e interfaz opcional de
            tokenizador.

    Returns:
        Chunks incluidos en el orden recibido, excluidos y metadata del conteo,
        incluyendo si es estimado y si las semillas exceden el presupuesto.

    Raises:
        ValueError: Si el presupuesto o conteo del tokenizador es inválido.
        TypeError: Si el tokenizador no cumple una interfaz callable o
            ``encode`` admitida.
    """
    budget = config.get("max_tokens")
    if budget is not None and (
        not isinstance(budget, int) or isinstance(budget, bool) or budget < 1
    ):
        raise ValueError("max_tokens debe ser un entero positivo o None.")
    counter = _token_counter(config)

    def count(text: str) -> int:
        return counter(text) if counter is not None else _estimated_token_count(text)

    if budget is None:
        total = count("\n\n".join(chunk.text for chunk in chunks))
        return chunks, [], {
            "token_count": total,
            "token_count_is_estimated": counter is None,
            "token_budget": None,
            "seed_budget_exceeded": False,
        }

    seeds = [chunk for chunk in chunks if chunk.role == "seed"]
    others = [chunk for chunk in chunks if chunk.role != "seed"]
    included: list[ContextChunk] = []
    excluded: list[ContextChunk] = []
    seed_used = count("\n\n".join(chunk.text for chunk in seeds))
    included.extend(seeds)

    for chunk in others:
        candidate_count = count("\n\n".join([item.text for item in included] + [chunk.text]))
        if candidate_count <= budget:
            included.append(chunk)
        else:
            excluded.append(chunk)

    included_ids = {chunk_identity(chunk) for chunk in included}
    included_in_original_order = [
        chunk for chunk in chunks if chunk_identity(chunk) in included_ids
    ]
    final_count = count("\n\n".join(chunk.text for chunk in included_in_original_order))
    return included_in_original_order, excluded, {
        "token_count": final_count,
        "token_count_is_estimated": counter is None,
        "token_budget": budget,
        "seed_budget_exceeded": seed_used > budget,
    }

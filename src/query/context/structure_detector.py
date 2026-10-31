"""Clasificación heurística basada en los campos disponibles del payload.

Si el texto y la metadata no permiten identificar una unidad con suficiente
confianza, el detector devuelve ``unknown``. Los patrones son señales para la
validación posterior; no sustituyen el análisis sintáctico del documento.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any


_ARTICLE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:art(?:ículo|iculo)?\.?)\s*"
    r"(\d+(?:\.\d+)*(?:[-–]\w+)?)\b",
    re.IGNORECASE,
)
_CHAPTER = re.compile(r"^\s*(?:cap[ií]tulo)\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
_SECTION = re.compile(r"^\s*(?:secci[oó]n|t[ií]tulo)\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
_NUMERAL = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?|[IVXLCDM]+[.)])\s+\S", re.IGNORECASE)
_LITERAL = re.compile(r"^\s*(?:[a-zñ][.)]|\([a-zñ]\))\s+\S", re.IGNORECASE)
_MARKDOWN_TITLE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_NOTE = re.compile(r"\bnota\s*:", re.IGNORECASE)
_REFORM = re.compile(
    r"\b(?:reform(?:a|ado|ada|atoria|atorio|as|as?))\b|"
    r"\bdisposici[oó]n\s+(?:reformatoria|reformada)\b",
    re.IGNORECASE,
)
_CONCORDANCE = re.compile(r"\bconcordancias?\s*:", re.IGNORECASE)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def detect_structure(payload: Mapping[str, Any]) -> dict[str, str | None]:
    """Clasifica un chunk usando primero el texto y luego metadata auxiliar.

    Se inspeccionan los primeros 1.200 caracteres del texto y todos los
    encabezados no vacíos. La estructura explícita al inicio del texto
    (artículo, capítulo o sección) prevalece sobre encabezados auxiliares.
    Concordancia requiere el marcador explícito en el texto principal; las
    referencias normativas aisladas y el marcador presente solo en
    ``headings`` no bastan. Después se evalúan notas de reforma, artículos y
    unidades estructurales identificables en encabezados, listas y títulos.
    Se preservan sufijos decimales o separados por guion de los artículos y se
    normalizan los acentos para comparar identificadores. Un encabezado o
    ``section_header`` sin identificador reconocible produce
    ``status="signal"``, salvo que sea únicamente el encabezado auxiliar
    ``Concordancias:``. Las páginas no se interpretan como identificadores.

    Args:
        payload: Mapping del payload Qdrant. Los valores opcionales ausentes o
            con tipo inesperado se tratan como no disponibles.

    Returns:
        Mapping con ``type``, ``id`` y ``status``. El estado puede ser
        ``identified``, ``signal`` o ``unknown``.
    """
    text = payload.get("text")
    text = text.strip() if isinstance(text, str) else ""
    headings = _strings(payload.get("headings"))
    item_types = {item.casefold() for item in _strings(payload.get("item_types"))}
    main_text = text[:1200]
    signals = "\n".join([*headings, main_text])
    first_line = text.splitlines()[0] if text else ""
    normalized_signal = normalize_text(signals)

    main_article = _ARTICLE.match(first_line)
    if main_article:
        return {
            "type": "article",
            "id": normalize_text(main_article.group(1)),
            "status": "identified",
        }

    for kind, pattern in (("chapter", _CHAPTER), ("section", _SECTION)):
        match = pattern.match(first_line)
        if match:
            return {"type": kind, "id": normalize_text(match.group(1)), "status": "identified"}

    if _CONCORDANCE.search(main_text):
        return {"type": "concordance", "id": None, "status": "identified"}

    if _REFORM.search(signals) and (
        _NOTE.search(signals) or "reformatoria" in normalized_signal or "reformada" in normalized_signal
    ):
        return {"type": "reform_note", "id": None, "status": "identified"}
    if _NOTE.search(first_line) or _NOTE.search(signals):
        return {"type": "complementary", "id": None, "status": "identified"}

    for candidate in headings:
        match = _ARTICLE.match(candidate)
        if match:
            return {"type": "article", "id": normalize_text(match.group(1)), "status": "identified"}
    for kind, pattern in (("chapter", _CHAPTER), ("section", _SECTION)):
        for candidate in headings:
            match = pattern.match(candidate)
            if match:
                return {"type": kind, "id": normalize_text(match.group(1)), "status": "identified"}

    if "list_item" in item_types or _LIST_PREFIX.match(first_line):
        return {"type": "list", "id": None, "status": "identified"}
    if _LITERAL.match(first_line):
        return {"type": "literal", "id": None, "status": "identified"}
    if _NUMERAL.match(first_line):
        return {"type": "numeral", "id": None, "status": "identified"}
    if any(_MARKDOWN_TITLE.match(heading) for heading in headings) or _MARKDOWN_TITLE.match(first_line):
        return {"type": "title", "id": normalize_text(first_line.lstrip("# ").strip()), "status": "identified"}
    if "section_header" in item_types:
        return {"type": "title", "id": None, "status": "signal"}
    if any(not _CONCORDANCE.match(heading) for heading in headings):
        return {"type": "title", "id": None, "status": "signal"}
    if text or headings:
        return {"type": "unknown", "id": None, "status": "unknown"}
    return {"type": "unknown", "id": None, "status": "unknown"}

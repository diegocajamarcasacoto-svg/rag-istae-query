"""Extracción y verificación automática de citas.

La respuesta del modelo cita unidades en línea (``[INT08-art12]``) y termina
con una sección "Fuentes:" con una cita textual por unidad. Aquí se comprueba,
sin intervención humana:

- que cada identificador citado exista en el corpus;
- que cada cita textual aparezca realmente en la unidad citada;
- si el modelo se abstuvo por falta de información.

Es una medida automática de trazabilidad y un indicador parcial de fidelidad;
no reemplaza la evaluación humana de exactitud.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

CITATION = re.compile(r"\[((?:INT|EXT)\d{2}-[A-Za-z0-9.\-]+)\]")
_SOURCES_HEADER = re.compile(r"^\s*(?:\*\*)?\s*fuentes\s*:?\s*(?:\*\*)?\s*:?\s*$", re.IGNORECASE | re.MULTILINE)
_SOURCE_LINE = re.compile(r"^\s*[-*]\s*\[(?P<id>[^\]]+)\]\s*(?P<quote>.*)$")
_ABSTENTION = re.compile(r"^\s*no encontr[ée] en la normativa disponible", re.IGNORECASE)
_MIN_QUOTE_CHARS = 12


@dataclass
class Citation:
    unit_id: str
    known: bool
    document_id: str | None
    quote: str | None = None
    quote_verified: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ParsedAnswer:
    body: str
    citations: list[Citation]
    abstained: bool
    unknown_ids: list[str]
    cited_without_source: list[str]

    @property
    def verified_quote_ratio(self) -> float | None:
        checked = [c for c in self.citations if c.quote_verified is not None]
        if not checked:
            return None
        return sum(1 for c in checked if c.quote_verified) / len(checked)


def normalize_for_match(text: str) -> str:
    """Normaliza para comparar citas: sin viñetas, marcas de formato, comillas ni tildes, espacios simples."""
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", " ", text)  # viñetas y numeración de listas
    # Marcadores de lista dentro de la frase ("…titulación: 1. Por una sola…", "a) Declaratoria"):
    # se quitan en la cita y en la unidad por igual, así la comparación es simétrica.
    text = re.sub(r"(?<!\S)(?:\d{1,2}[.)]|[a-zñ]\))(?=\s)", " ", text, flags=re.IGNORECASE)
    text = text.replace("**", "").replace("«", "").replace("»", "").replace("“", "").replace("”", "").replace('"', "")
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    text = re.sub(r"[^\w%$.,;:()/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .;:,")


def _quote_in_unit(quote: str, unit_text: str) -> bool | None:
    cleaned = normalize_for_match(quote)
    cleaned = re.sub(r"^\.\.\.|\.\.\.$", "", cleaned).strip()
    if len(cleaned) < _MIN_QUOTE_CHARS:
        return None
    unit = normalize_for_match(unit_text)
    if cleaned in unit:
        return True
    # Tolera elipsis internas: todas las partes deben aparecer en orden.
    parts = [part.strip() for part in re.split(r"\s*\.\.\.\s*|\s*…\s*", cleaned) if len(part.strip()) >= 6]
    position = 0
    for part in parts:
        found = unit.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return bool(parts)


def parse_answer(text: str, units: dict[str, tuple[str, str]]) -> ParsedAnswer:
    """Separa cuerpo y fuentes y verifica cada cita.

    ``units`` mapea ``unit_id`` → (``document_id``, texto de la unidad).
    """
    header = _SOURCES_HEADER.search(text)
    body = text[: header.start()].rstrip() if header else text.strip()
    sources_block = text[header.end():] if header else ""

    quotes: dict[str, str] = {}
    for line in sources_block.splitlines():
        match = _SOURCE_LINE.match(line)
        if match:
            quotes.setdefault(match.group("id").strip(), match.group("quote").strip())

    ordered_ids = list(dict.fromkeys([*CITATION.findall(body), *quotes.keys()]))
    citations: list[Citation] = []
    for unit_id in ordered_ids:
        known = unit_id in units
        document_id, unit_text = units.get(unit_id, (None, ""))
        quote = quotes.get(unit_id)
        verified = _quote_in_unit(quote, unit_text) if known and quote else None
        citations.append(Citation(unit_id, known, document_id, quote, verified))

    body_ids = set(CITATION.findall(body))
    return ParsedAnswer(
        body=body,
        citations=citations,
        abstained=bool(_ABSTENTION.match(text)),
        unknown_ids=[c.unit_id for c in citations if not c.known],
        cited_without_source=sorted(body_ids - set(quotes)),
    )

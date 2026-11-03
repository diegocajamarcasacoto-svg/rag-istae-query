"""Estructuración de una norma en unidades citables con IDs estables.

Dos modos, elegidos automáticamente por documento:

- ``articulado``: la norma numera artículos desde 1. Cada artículo es una
  unidad (``INT08-art12``). Las disposiciones generales, transitorias, etc.
  son unidades propias (``INT08-disp-transitoria-primera``).
- ``secciones``: la norma no tiene artículos propios (guías, modelos,
  normativas por apartados). Cada encabezado inicia una sección
  (``INT09-sec03``).

Para no confundir artículos citados de otras normas con artículos propios, en
modo articulado solo se aceptan encabezados que siguen la numeración: el
primero debe ser el 1, y los siguientes deben avanzar como máximo
``MAX_ARTICLE_GAP`` posiciones (o reiniciar en 1, como en un anexo). Los
candidatos rechazados se reportan para revisión manual.

El texto no se reescribe: las unidades conservan las líneas originales en
orden, de modo que concatenarlas reproduce el documento limpio.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from query.long_context.models import ParsedDocument, StructureMode, Unit, UnitKind

MAX_ARTICLE_GAP = 5
MIN_ARTICLES_FOR_ARTICULATED = 3

_ARTICLE = re.compile(
    r"^\s*(?:[-*>]\s+)?(?:#{1,6}\s*)?(?:\*\*)?\s*(?:art[íi]culo|art\.)\s*"
    r"(?P<num>\d+(?:\.\d+)?(?:[-–]\s*[a-z]\b|\s+(?:bis|ter|quater)\b)?|innumerado|[úu]nico)",
    re.IGNORECASE,
)
_ORDINAL_WORDS = (
    r"PRIMER[AO]|SEGUND[AO]|TERCER[AO]|CUART[AO]|QUINT[AO]|SEXT[AO]|S[ÉE]PTIM[AO]|"
    r"OCTAV[AO]|NOVEN[AO]"
)
_ORDINAL = re.compile(
    rf"^\s*(?:[-*]\s+)?(?:#{{1,6}}\s*)?(?:\*\*)?\s*(?P<ord>"
    rf"D[ÉE]CIM[AO](?:\s+(?:{_ORDINAL_WORDS}))?|UND[ÉE]CIM[AO]|DUOD[ÉE]CIM[AO]|"
    rf"VIG[ÉE]SIM[AO](?:\s+(?:{_ORDINAL_WORDS}))?|{_ORDINAL_WORDS}|[ÚU]NIC[AO])"
    rf"\b\s*(?:\*\*)?\s*[.\-–:,]",
    re.IGNORECASE,
)
_DISPOSITIONS = re.compile(
    r"DISPOSICI[ÓO]N(?:ES)?\s+(?P<tipo>GENERAL|TRANSITORIA|DEROGATORIA|FINAL|REFORMATORIA)",
    re.IGNORECASE,
)
_STRUCTURAL = re.compile(
    r"^(?P<kind>T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|PAR[ÁA]GRAFO)\b", re.IGNORECASE
)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<t>.+?)\s*#*\s*$")
_BOLD_LINE = re.compile(r"^\s*\*\*(?P<t>[^*]{2,160})\*\*\s*[:.]?\s*$")
_MAX_PLAIN_HEADING = 120


def _plain(line: str) -> str:
    text = re.sub(r"^\s*(?:[-*>]\s+)?#{0,6}\s*", "", line)
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def _ascii_slug(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9.]+", "-", ascii_text).strip("-")


def _article_number(raw: str) -> tuple[str, int | None]:
    slug = _ascii_slug(raw.replace("–", "-"))
    main = re.match(r"\d+", slug)
    return slug, int(main.group()) if main else None


def _article_title(line: str) -> str:
    plain = _plain(line)
    match = _ARTICLE.match(plain)
    rest = plain[match.end():] if match else plain
    rest = rest.lstrip(" .,:;-–—")
    title = re.split(r"\.\s*[-–—]|\.\s|:\s|\s[-–—]\s", rest, maxsplit=1)[0]
    return title.strip(" .-–—:")[:_MAX_PLAIN_HEADING]


def _heading_text(line: str) -> str | None:
    for pattern in (_MD_HEADING, _BOLD_LINE):
        match = pattern.match(line)
        if match:
            return _plain(match.group("t"))
    plain = line.strip()
    if plain and len(plain) <= _MAX_PLAIN_HEADING and _STRUCTURAL.match(plain):
        return _plain(plain)
    return None


def _accepted_articles(lines: list[str]) -> tuple[dict[int, str], list[str], list[str]]:
    """Devuelve {índice de línea: número}, candidatos rechazados y saltos de numeración.

    Un salto mayor que ``MAX_ARTICLE_GAP`` se acepta solo si el candidato
    siguiente continúa la nueva numeración (por ejemplo, 55 → 61 → 62): así se
    distingue un tramo de artículos ausente en el texto de una cita aislada de
    otra norma (55 → 95 → 56).
    """
    candidates = []
    for index, line in enumerate(lines):
        match = _ARTICLE.match(line)
        if match:
            slug, number = _article_number(match.group("num"))
            candidates.append((index, slug, number))
    numeric = [number for _, _, number in candidates]
    if len(candidates) < MIN_ARTICLES_FOR_ARTICULATED or 1 not in numeric:
        return {}, [lines[i].strip()[:100] for i, _, _ in candidates], []

    accepted: dict[int, str] = {}
    rejected: list[str] = []
    jumps: list[str] = []
    last: int | None = None
    for position, (index, slug, number) in enumerate(candidates):
        if number is None:  # artículo innumerado o único
            accepted[index] = slug
            continue
        if last is None:
            ok = number == 1
        elif number == 1 or last <= number <= last + MAX_ARTICLE_GAP:
            ok = True
        elif number > last + MAX_ARTICLE_GAP:
            following = next((n for _, _, n in candidates[position + 1:] if n is not None), None)
            ok = following is not None and number < following <= number + MAX_ARTICLE_GAP
            if ok:
                jumps.append(f"salto de numeración {last} → {number}")
        else:
            ok = False
        if ok:
            accepted[index] = slug
            last = number
        else:
            rejected.append(lines[index].strip()[:100])
    return accepted, rejected, jumps


@dataclass
class _Builder:
    document_id: str
    units: list[Unit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    used_ids: dict[str, int] = field(default_factory=dict)
    pending_headings: list[str] = field(default_factory=list)
    body_lines: int = 0

    def start(self, kind: UnitKind, local_id: str, label: str, title: str, path: str, first_line: str | None) -> None:
        unit_id = f"{self.document_id}-{local_id}"
        count = self.used_ids.get(unit_id, 0) + 1
        self.used_ids[unit_id] = count
        if count > 1:
            self.warnings.append(f"ID repetido {unit_id}: se asignó {unit_id}-{count}")
            unit_id = f"{unit_id}-{count}"
        lines = list(self.pending_headings)
        if first_line is not None:
            lines.append(first_line)
        self.pending_headings.clear()
        self.units.append(Unit(unit_id, kind, label, title, path, "\n".join(lines)))
        self.body_lines = 0

    def append(self, line: str) -> None:
        unit = self.units[-1]
        if not line.strip() and self.pending_headings:
            # Una línea en blanco tras un encabezado no decide a qué unidad pertenece.
            return
        if self.pending_headings:
            unit.text = "\n".join([unit.text, *self.pending_headings]) if unit.text else "\n".join(self.pending_headings)
            self.pending_headings.clear()
        unit.text = f"{unit.text}\n{line}" if unit.text else line
        if line.strip():
            self.body_lines += 1

    def finish(self) -> list[Unit]:
        if self.pending_headings:
            self.append("")
        for unit in self.units:
            unit.text = unit.text.strip()
        return [unit for unit in self.units if unit.text]


def parse_document(document_id: str, text: str) -> ParsedDocument:
    """Divide el texto limpio de una norma en unidades citables."""
    lines = text.split("\n")
    article_lines, rejected, jumps = _accepted_articles(lines)
    mode: StructureMode = "articulado" if article_lines else "secciones"

    builder = _Builder(document_id)
    builder.start("preambulo", "preambulo", "Preámbulo", "", "", None)
    path_parts: dict[str, str] = {}
    disposition_type: str | None = None
    section_number = 0

    def current_path() -> str:
        order = ("titulo", "capitulo", "seccion", "disposiciones")
        return " › ".join(path_parts[key] for key in order if key in path_parts)

    for index, line in enumerate(lines):
        if index in article_lines:
            disposition_type = None
            path_parts.pop("disposiciones", None)
            number = article_lines[index]
            builder.start("articulo", f"art{number}", f"Art. {number}", _article_title(line), current_path(), line)
            continue

        heading = _heading_text(line)
        dispositions = _DISPOSITIONS.search(line) if len(line.strip()) <= _MAX_PLAIN_HEADING else None
        if dispositions and (heading or len(_plain(line)) <= 60):
            disposition_type = _ascii_slug(dispositions.group("tipo"))
            path_parts["disposiciones"] = _plain(line)
            builder.pending_headings.append(line)
            continue

        if disposition_type:
            ordinal = _ORDINAL.match(line)
            if ordinal:
                word = ordinal.group("ord")
                builder.start(
                    "disposicion",
                    f"disp-{disposition_type}-{_ascii_slug(word)}",
                    f"Disposición {disposition_type} {word.capitalize()}",
                    "",
                    current_path(),
                    line,
                )
                continue

        if heading is not None:
            structural = _STRUCTURAL.match(heading)
            if structural:
                kind = _ascii_slug(structural.group("kind")).replace("parrafo", "seccion")
                kind = {"titulo": "titulo", "capitulo": "capitulo"}.get(kind, "seccion")
                path_parts[kind] = heading
                if kind == "titulo":
                    path_parts.pop("capitulo", None)
                    path_parts.pop("seccion", None)
                elif kind == "capitulo":
                    path_parts.pop("seccion", None)
            if mode == "secciones":
                last = builder.units[-1]
                if last.kind == "seccion" and builder.body_lines == 0:
                    last.title = f"{last.title} › {heading}" if last.title else heading
                    last.text = f"{last.text}\n{line}"
                else:
                    section_number += 1
                    builder.start("seccion", f"sec{section_number:02d}", f"Sección {section_number}", heading, current_path(), line)
                continue
            if structural:
                builder.pending_headings.append(line)
                continue

        builder.append(line)

    units = builder.finish()
    warnings = list(builder.warnings)
    if jumps:
        warnings.append(
            "Artículos posiblemente ausentes en la extracción (verificar contra el original): " + "; ".join(jumps)
        )
    if rejected and mode == "articulado":
        warnings.append(f"{len(rejected)} encabezados de artículo fuera de secuencia (posibles citas de otras normas)")
    return ParsedDocument(document_id, mode, units, warnings, rejected if mode == "articulado" else [])

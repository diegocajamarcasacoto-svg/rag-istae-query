"""Limpieza conservadora del Markdown exportado por Docling.

Solo elimina ruido que no aporta contenido normativo:

- marcadores de imagen vacíos (``<!-- image -->``);
- repeticiones de líneas de membrete de página (UBICACIÓN:, VERSIÓN:, CÓDIGO:,
  ÁREA:, FECHA:, PÁGINA). Se conserva la primera aparición de cada una;
- líneas que solo contienen un número de página;
- espacios al final de línea y bloques de líneas vacías.

No se eliminan líneas de contenido aunque se repitan (por ejemplo, "Tipo de
indicador: Cualitativo" en las guías del CACES es información de cada
indicador).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_IMAGE = re.compile(r"^\s*<!--\s*image\s*-->\s*$", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
_LETTERHEAD = re.compile(
    r"^\s*(?:\*\*)?\s*(?:UBICACI[ÓO]N|VERSI[ÓO]N|C[ÓO]DIGO|[ÁA]REA|FECHA|P[ÁA]GINA)\s*(?:\*\*)?\s*:",
    re.IGNORECASE,
)
_MAX_LETTERHEAD_LENGTH = 100


@dataclass
class CleaningReport:
    image_placeholders: int = 0
    page_numbers: int = 0
    repeated_letterhead: Counter[str] = field(default_factory=Counter)

    @property
    def removed_lines(self) -> int:
        return self.image_placeholders + self.page_numbers + sum(self.repeated_letterhead.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "image_placeholders": self.image_placeholders,
            "page_numbers": self.page_numbers,
            "repeated_letterhead": dict(self.repeated_letterhead),
            "removed_lines": self.removed_lines,
        }


def clean_markdown(text: str) -> tuple[str, CleaningReport]:
    """Devuelve el texto limpio y un informe de lo eliminado."""
    report = CleaningReport()
    seen_letterhead: set[str] = set()
    kept: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.rstrip()
        if _IMAGE.match(line):
            report.image_placeholders += 1
            continue
        if _PAGE_NUMBER.match(line):
            report.page_numbers += 1
            continue
        stripped = line.strip()
        if len(stripped) <= _MAX_LETTERHEAD_LENGTH and _LETTERHEAD.match(stripped):
            key = stripped.casefold()
            if key in seen_letterhead:
                report.repeated_letterhead[stripped] += 1
                continue
            seen_letterhead.add(key)
        kept.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned, report

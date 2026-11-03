"""Catálogo normativo: fichas de documentos, rangos jerárquicos y relaciones.

El catálogo se genera una vez desde la Matriz de reglamentos (Excel) y se
guarda como JSON versionable (``config/catalogo_normativo.json``), para que
el equipo pueda revisarlo y corregirlo en git.

Los rangos jerárquicos son una PROPUESTA basada en el art. 425 de la
Constitución (ley orgánica > reglamentos y acuerdos > resoluciones) y en el
tipo de documento. Deben ser validados por una persona con criterio jurídico
antes de la evaluación final; por eso el JSON registra ``rank_status``.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from query.long_context.models import DocumentInfo, Relation

RANK_LABELS: dict[int, str] = {
    1: "Constitución",
    2: "Ley orgánica",
    3: "Reglamento o acuerdo de organismo nacional",
    4: "Instrumento técnico de organismo nacional",
    5: "Estatuto institucional",
    6: "Reglamento o normativa interna",
}
RANK_STATUS = "propuesta: pendiente de validación jurídica"
UNKNOWN_RANK = 99

_DOC_ID = re.compile(r"^\s*((?:INT|EXT)\d{2})\b", re.IGNORECASE)


def propose_rank(doc_type: str, issuer: str, name: str, code: str) -> int:
    """Asigna un rango jerárquico propuesto a partir de la ficha del documento."""
    kind = doc_type.strip().casefold()
    lowered = name.casefold()
    if kind.startswith("ext"):
        if "ley orgánica" in lowered or "ley organica" in lowered:
            return 2
        if "caces" in issuer.casefold():
            return 4
        return 3
    if code.upper().endswith("-EST") or "estatuto" in lowered:
        return 5
    return 6


def document_id_from_file(file_name: str) -> str | None:
    match = _DOC_ID.match(file_name or "")
    return match.group(1).upper() if match else None


def _cell(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _find_header(row: tuple[Any, ...], *required: str) -> dict[str, int] | None:
    names = {_cell(value).casefold(): index for index, value in enumerate(row) if _cell(value)}
    if all(name.casefold() in names for name in required):
        return names
    return None


def build_catalog_from_matriz(xlsx_path: Path) -> dict[str, Any]:
    """Lee la Matriz de reglamentos y devuelve el catálogo como diccionario."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError("Instale openpyxl para leer la Matriz: pip install openpyxl") from exc

    xlsx_path = Path(xlsx_path)
    workbook = load_workbook(xlsx_path, read_only=True, data_only=True)
    documents: list[dict[str, Any]] = []
    code_to_id: dict[str, str] = {}

    sheet = workbook.worksheets[0]
    header: dict[str, int] | None = None
    for row in sheet.iter_rows(values_only=True):
        found = _find_header(row, "Código", "Archivo en esta carpeta")
        if found:
            header = found
            continue
        if not header:
            continue
        doc_type = _cell(row[header["tipo"]]) if "tipo" in header else ""
        if doc_type.casefold() not in {"interno", "externo"}:
            continue
        file_name = _cell(row[header["archivo en esta carpeta"]])
        document_id = document_id_from_file(file_name)
        if not document_id:
            continue
        code = _cell(row[header["código"]])
        name = _cell(row[header["reglamento / norma"]])
        issuer = _cell(row[header["órgano emisor"]])
        rank = propose_rank(doc_type, issuer, name, code)
        code_to_id[code] = document_id
        documents.append({
            "document_id": document_id,
            "code": code,
            "name": name,
            "doc_type": doc_type,
            "issuer": issuer,
            "domain": _cell(row[header["dominio de la tesis"]]),
            "rank": rank,
            "rank_label": RANK_LABELS[rank],
            "source_file": file_name,
        })

    relations: list[dict[str, Any]] = []
    if len(workbook.worksheets) > 1:
        rel_header: dict[str, int] | None = None
        for row in workbook.worksheets[1].iter_rows(values_only=True):
            found = _find_header(row, "Código origen", "Código destino")
            if found:
                rel_header = found
                continue
            if not rel_header:
                continue
            source_code = _cell(row[rel_header["código origen"]])
            target_code = _cell(row[rel_header["código destino"]])
            relation = _cell(row[rel_header["relación"]]) if "relación" in rel_header else ""
            if not source_code or not target_code or not relation:
                continue
            relations.append({
                "source_document_id": code_to_id.get(source_code),
                "source_code": source_code,
                "relation": relation,
                "target_document_id": code_to_id.get(target_code),
                "target_code": target_code,
                "verification": _cell(row[rel_header["verificación"]]) if "verificación" in rel_header else "",
            })
    workbook.close()

    documents.sort(key=lambda item: item["document_id"])
    return {
        "source": xlsx_path.name,
        "source_sha256": hashlib.sha256(xlsx_path.read_bytes()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rank_labels": {str(key): value for key, value in RANK_LABELS.items()},
        "rank_status": RANK_STATUS,
        "documents": documents,
        "relations": relations,
    }


def load_catalog(path: Path) -> tuple[dict[str, DocumentInfo], list[Relation], dict[str, Any]]:
    """Carga el catálogo JSON y devuelve fichas por documento, relaciones y metadatos."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = {item["document_id"]: DocumentInfo(**item) for item in data.get("documents", [])}
    relations = [Relation(**item) for item in data.get("relations", [])]
    meta = {key: value for key, value in data.items() if key not in {"documents", "relations"}}
    return documents, relations, meta


def fallback_document(document_id: str, source_file: str) -> DocumentInfo:
    """Ficha mínima para un documento que no figura en el catálogo."""
    name = re.sub(r"^\s*(?:INT|EXT)\d{2}\s*-\s*", "", Path(source_file).stem)
    return DocumentInfo(
        document_id=document_id,
        code=document_id,
        name=name,
        doc_type="",
        issuer="",
        domain="",
        rank=UNKNOWN_RANK,
        rank_label="Sin clasificar",
        source_file=source_file,
    )

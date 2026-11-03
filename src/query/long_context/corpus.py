"""Construcción de los corpus de contexto largo (variantes A y A+J).

Fuente: la salida de la ingesta (``outputs/extracted_text/<sha256>.txt`` más
``outputs/document_mapping.jsonl``), es decir, la misma extracción con Docling
que usa el RAG, sin fragmentación ni recuperación.

- Variante A: documentos completos en orden neutro (por código), con un
  encabezado por norma y la etiqueta de cada unidad citable.
- Variante A+J: los mismos documentos ordenados por rango jerárquico, con el
  rango en cada encabezado, la escala jerárquica al inicio y la lista de
  relaciones entre normas al final.

Ambas variantes contienen exactamente el mismo texto normativo; solo cambian el
orden y la información jerárquica añadida.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from query.long_context.catalog import fallback_document
from query.long_context.cleaning import CleaningReport, clean_markdown
from query.long_context.models import DocumentInfo, ParsedDocument, Relation, Unit
from query.long_context.structure import parse_document

Variant = Literal["A", "AJ"]


@dataclass
class CorpusDocument:
    info: DocumentInfo
    parsed: ParsedDocument
    cleaning: CleaningReport
    source_sha256: str
    raw_chars: int
    clean_chars: int
    warnings: list[str] = field(default_factory=list)


def read_mapping(outputs_dir: Path) -> list[dict[str, str]]:
    path = Path(outputs_dir) / "document_mapping.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(rows, key=lambda row: row["document_id"])


def load_corpus(outputs_dir: Path, catalog: dict[str, DocumentInfo]) -> list[CorpusDocument]:
    """Lee, limpia y estructura todos los documentos de la ingesta."""
    outputs_dir = Path(outputs_dir)
    documents: list[CorpusDocument] = []
    for row in read_mapping(outputs_dir):
        document_id = row["document_id"]
        text_path = outputs_dir / "extracted_text" / f"{row['source_sha256']}.txt"
        raw = text_path.read_text(encoding="utf-8")
        cleaned, report = clean_markdown(raw)
        info = catalog.get(document_id) or fallback_document(document_id, row["relative_path"])
        warnings: list[str] = []
        if document_id not in catalog:
            warnings.append("No figura en el catálogo normativo: se usó una ficha mínima sin rango.")
        parsed = parse_document(document_id, cleaned)
        warnings.extend(parsed.warnings)
        documents.append(CorpusDocument(info, parsed, report, row["source_sha256"], len(raw), len(cleaned), warnings))
    return documents


def render_unit(unit: Unit) -> str:
    return f"[{unit.unit_id}] {unit.text}"


def _document_header(info: DocumentInfo, variant: Variant) -> str:
    if variant == "AJ":
        return (
            f"=== [{info.document_id}] {info.name} · Jerarquía {info.rank}: {info.rank_label}"
            f" · Emisor: {info.issuer or 'no registrado'} ==="
        )
    return f"=== [{info.document_id}] {info.name} ==="


def _hierarchy_preamble(rank_labels: dict[str, str]) -> str:
    lines = ["=== ESCALA DE JERARQUÍA NORMATIVA (1 = mayor rango) ==="]
    lines += [f"{key}. {label}" for key, label in sorted(rank_labels.items(), key=lambda item: int(item[0]))]
    lines.append("Los documentos aparecen ordenados de mayor a menor jerarquía.")
    return "\n".join(lines)


def _relations_block(relations: list[Relation], names: dict[str, str]) -> str:
    lines = ["=== RELACIONES ENTRE NORMAS ==="]
    for rel in relations:
        source = f"[{rel.source_document_id}] {names.get(rel.source_document_id or '', rel.source_code)}" if rel.source_document_id else rel.source_code
        target = f"[{rel.target_document_id}] {names.get(rel.target_document_id or '', rel.target_code)}" if rel.target_document_id else rel.target_code
        lines.append(f"- {source} — {rel.relation} → {target}")
    return "\n".join(lines)


def render_corpus(
    documents: list[CorpusDocument],
    variant: Variant,
    relations: list[Relation] | None = None,
    rank_labels: dict[str, str] | None = None,
) -> str:
    """Devuelve el texto completo que se enviará al LLM como contexto."""
    if variant == "AJ":
        ordered = sorted(documents, key=lambda doc: (doc.info.rank, doc.info.document_id))
    else:
        ordered = sorted(documents, key=lambda doc: doc.info.document_id)
    blocks: list[str] = []
    if variant == "AJ":
        blocks.append(_hierarchy_preamble(rank_labels or {}))
    for doc in ordered:
        body = "\n\n".join(render_unit(unit) for unit in doc.parsed.units)
        blocks.append(f"{_document_header(doc.info, variant)}\n{body}")
    if variant == "AJ" and relations:
        names = {doc.info.document_id: doc.info.name for doc in documents}
        blocks.append(_relations_block(relations, names))
    return "\n\n".join(blocks) + "\n"


def units_records(documents: list[CorpusDocument]) -> list[dict[str, Any]]:
    """Registros por unidad citable (insumo reutilizable también por el RAG)."""
    records = []
    for doc in documents:
        for order, unit in enumerate(doc.parsed.units, 1):
            records.append({
                **unit.to_dict(),
                "document_id": doc.info.document_id,
                "document_name": doc.info.name,
                "rank": doc.info.rank,
                "rank_label": doc.info.rank_label,
                "issuer": doc.info.issuer,
                "order": order,
                "source_sha256": doc.source_sha256,
                "char_count": len(unit.text),
            })
    return records


def build_report(documents: list[CorpusDocument], corpora: dict[str, str]) -> dict[str, Any]:
    """Informe de construcción: tamaños, estructura detectada y advertencias."""
    per_document = []
    for doc in documents:
        parsed = doc.parsed
        per_document.append({
            "document_id": doc.info.document_id,
            "name": doc.info.name,
            "rank": doc.info.rank,
            "mode": parsed.mode,
            "articles": parsed.count("articulo"),
            "sections": parsed.count("seccion"),
            "dispositions": parsed.count("disposicion"),
            "raw_chars": doc.raw_chars,
            "clean_chars": doc.clean_chars,
            "cleaning": doc.cleaning.to_dict(),
            "rejected_article_candidates": parsed.rejected_article_candidates[:10],
            "warnings": doc.warnings,
        })
    corpus_stats = {}
    for name, text in corpora.items():
        chars = len(text)
        corpus_stats[name] = {
            "chars": chars,
            "utf8_bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "tokens_estimate_4_chars": round(chars / 4),
            "tokens_estimate_3_5_chars": round(chars / 3.5),
            "note": "Estimación. El conteo real requiere el tokenizador del modelo elegido.",
        }
    return {
        "documents": len(documents),
        "units": sum(len(doc.parsed.units) for doc in documents),
        "raw_chars": sum(doc.raw_chars for doc in documents),
        "clean_chars": sum(doc.clean_chars for doc in documents),
        "corpora": corpus_stats,
        "per_document": per_document,
    }

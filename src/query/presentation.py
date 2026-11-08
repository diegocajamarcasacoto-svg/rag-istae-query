"""Presentación del contexto por documentos, en el formato de ``rag-istae-api``.

Reescribe el texto de un ``ContextResult`` por unidades (``--expansion units``)
con el mismo formato con que la API presenta el corpus en A/AJ, para que las
arquitecturas B/C usen las mismas reglas de cita y la misma verificación:

    === [INT08] Reglamento de Titulación y Graduación ===
    [INT08-art12] Artículo 12. ...

- ``hybrid`` (B): documentos en orden de relevancia (el del mejor artículo
  primero); dentro de cada documento, artículos en orden de relevancia.
- ``hierarchical`` (C): como AJ, escala jerárquica al inicio, documentos por
  rango (mayor jerarquía primero) con rango y emisor en el encabezado, y al
  final las relaciones del catálogo **entre las normas presentes**. Las normas
  que solo aportan unidades de jerarquía se marcan «incluida por jerarquía».

No cambia qué unidades se envían ni su texto; solo el armado del contexto.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from query.context.models import ContextChunk, ContextResult
from query.hierarchy.catalog import Catalog
from query.units import estimate_tokens

PRESENTATION = "documents"
INCLUDED_BY_HIERARCHY = "incluida por jerarquía"


def unit_body(chunk: ContextChunk) -> str:
    """Texto de la unidad sin la línea de encabezado que agrega ``build_unit_context``."""
    header, _, body = chunk.text.partition("\n")
    if chunk.chunk_id and header.startswith(f"[{chunk.chunk_id}]"):
        return body.strip()
    return chunk.text.strip()


def _hierarchy_preamble(catalog: Catalog) -> str:
    lines = ["=== ESCALA DE JERARQUÍA NORMATIVA (1 = mayor rango) ==="]
    lines += [f"{key}. {label}" for key, label in sorted(catalog.rank_labels.items(), key=lambda item: int(item[0]))]
    lines.append("Los documentos aparecen ordenados de mayor a menor jerarquía.")
    return "\n".join(lines)


def _relations_block(catalog: Catalog, document_ids: list[str]) -> str | None:
    present = set(document_ids)
    names = {doc: (catalog.document(doc).name if catalog.document(doc) else doc) for doc in present}
    lines = [
        f"- [{rel.source_document_id}] {names[rel.source_document_id]} — {rel.relation} → "
        f"[{rel.target_document_id}] {names[rel.target_document_id]}"
        for rel in catalog.relations
        if rel.source_document_id in present and rel.target_document_id in present
    ]
    return "\n".join(["=== RELACIONES ENTRE NORMAS ===", *lines]) if lines else None


def render_documents(result: ContextResult, catalog: Catalog | None) -> ContextResult:
    """Devuelve el mismo resultado con ``context`` presentado por documentos.

    Requiere un contexto por unidades. Sin catálogo (solo admitido en
    ``hybrid``) los documentos se nombran por su ``document_id``.
    """
    if result.metadata.get("expansion") != "units":
        raise ValueError("La presentación por documentos requiere --expansion units.")
    hierarchical = result.metadata.get("retrieval_mode") == "hierarchical"
    if hierarchical and catalog is None:
        raise ValueError("La presentación jerárquica requiere el catálogo normativo.")

    groups: dict[str, list[ContextChunk]] = {}
    for chunk in result.chunks:  # ya vienen en orden de relevancia (base) y luego jerarquía
        groups.setdefault(chunk.document_id or "SIN-DOCUMENTO", []).append(chunk)

    first_seen = {doc: position for position, doc in enumerate(groups)}
    if hierarchical:
        def rank(doc: str) -> int:
            info = catalog.document(doc)
            return info.rank if info else 99
        order = sorted(groups, key=lambda doc: (rank(doc), first_seen[doc]))
    else:
        order = list(groups)

    warnings = []
    blocks: list[str] = []
    if hierarchical:
        blocks.append(_hierarchy_preamble(catalog))
    for doc in order:
        info = catalog.document(doc) if catalog else None
        if info is None and catalog is not None:
            warnings.append(f"{doc} no figura en el catálogo: se nombra por su código.")
        name = info.name if info else doc
        if hierarchical:
            rank_text = f"Jerarquía {info.rank}: {info.rank_label}" if info else "Jerarquía sin clasificar"
            header = f"=== [{doc}] {name} · {rank_text} · Emisor: {(info.issuer if info else '') or 'no registrado'}"
            if all(chunk.role == "hierarchy" for chunk in groups[doc]):
                header += f" · {INCLUDED_BY_HIERARCHY}"
            header += " ==="
        else:
            header = f"=== [{doc}] {name} ==="
        units = "\n\n".join(f"[{chunk.chunk_id}] {unit_body(chunk)}" for chunk in groups[doc])
        blocks.append(f"{header}\n{units}")
    if hierarchical:
        relations = _relations_block(catalog, order)
        if relations:
            blocks.append(relations)

    context = "\n\n".join(blocks)
    metrics: dict[str, Any] = dict(result.metadata.get("metrics") or {})
    metrics["token_count"] = estimate_tokens(context)
    metrics["token_count_is_estimated"] = True
    metadata = {
        **result.metadata,
        "presentation": PRESENTATION,
        "metrics": metrics,
        "document_order": order,
        "warnings": [*(result.metadata.get("warnings") or []), *warnings],
    }
    return replace(result, context=context, metadata=metadata)

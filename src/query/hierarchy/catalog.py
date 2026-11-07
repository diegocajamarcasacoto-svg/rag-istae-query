"""Catálogo normativo: rango jerárquico de cada norma y relaciones entre normas.

Es una copia de ``catalogo_normativo.json`` de ``rag-istae-api`` (generado desde
``Matriz_Reglamentos_TESIS_ISTAE_v5.xlsx``), importada a ``storage/catalog`` con
``scripts/import_catalog.py``. Rangos 1–6, donde 1 es el mayor; en la matriz
figuran como «propuesta: pendiente de validación jurídica».
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RELATION_DEVELOPS = "Desarrolla"
RELATION_COMPATIBLE = "Debe ser compatible con"
RELATION_ALIASES = {
    "desarrolla": RELATION_DEVELOPS,
    "compatible": RELATION_COMPATIBLE,
}


@dataclass(frozen=True)
class CatalogDocument:
    document_id: str
    code: str
    name: str
    rank: int
    rank_label: str
    issuer: str = ""


@dataclass(frozen=True)
class CatalogRelation:
    source_document_id: str
    relation: str
    target_document_id: str
    verification: str = ""


@dataclass(frozen=True)
class Catalog:
    documents: dict[str, CatalogDocument]
    relations: tuple[CatalogRelation, ...]
    rank_labels: dict[str, str]
    source: str = ""
    source_sha256: str = ""

    def document(self, document_id: str) -> CatalogDocument | None:
        return self.documents.get(document_id)

    def outgoing(self, document_id: str, relations: Iterable[str] | None = None) -> list[CatalogRelation]:
        allowed = set(relations) if relations is not None else None
        return [
            rel for rel in self.relations
            if rel.source_document_id == document_id and (allowed is None or rel.relation in allowed)
        ]


def resolve_relations(names: Iterable[str]) -> tuple[str, ...]:
    """Convierte alias de línea de comandos (``desarrolla``, ``compatible``) en relaciones del catálogo."""
    resolved = []
    for name in names:
        key = name.strip()
        if not key:
            continue
        relation = RELATION_ALIASES.get(key.lower(), key)
        if relation not in (RELATION_DEVELOPS, RELATION_COMPATIBLE):
            raise ValueError(f"Relación desconocida: {name!r}. Use: desarrolla, compatible.")
        if relation not in resolved:
            resolved.append(relation)
    if not resolved:
        raise ValueError("Indique al menos una relación.")
    return tuple(resolved)


def catalog_from_dict(data: dict[str, Any]) -> Catalog:
    documents = {}
    for item in data.get("documents") or []:
        document = CatalogDocument(
            document_id=item["document_id"],
            code=item.get("code") or item["document_id"],
            name=item.get("name") or item["document_id"],
            rank=int(item["rank"]),
            rank_label=item.get("rank_label") or "",
            issuer=item.get("issuer") or "",
        )
        if document.document_id in documents:
            raise ValueError(f"document_id duplicado en el catálogo: {document.document_id}")
        documents[document.document_id] = document
    if not documents:
        raise ValueError("El catálogo no contiene documentos.")
    relations = tuple(
        CatalogRelation(
            source_document_id=item["source_document_id"],
            relation=item["relation"],
            target_document_id=item["target_document_id"],
            verification=item.get("verification") or "",
        )
        for item in data.get("relations") or []
        # Relaciones con normas fuera del corpus (sin document_id) no son navegables.
        if item.get("source_document_id") and item.get("target_document_id")
    )
    return Catalog(
        documents=documents,
        relations=relations,
        rank_labels={str(key): value for key, value in (data.get("rank_labels") or {}).items()},
        source=data.get("source") or "",
        source_sha256=data.get("source_sha256") or "",
    )


def load_catalog(path: Path) -> Catalog:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Impórtelo con scripts/import_catalog.py."
        )
    return catalog_from_dict(json.loads(path.read_text(encoding="utf-8")))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

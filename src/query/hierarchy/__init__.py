"""Recuperación jerárquica (arquitectura C): catálogo normativo y expansión a normas superiores."""

from query.hierarchy.catalog import (
    RELATION_COMPATIBLE,
    RELATION_DEVELOPS,
    Catalog,
    CatalogDocument,
    CatalogRelation,
    catalog_from_dict,
    load_catalog,
    resolve_relations,
)
from query.hierarchy.expansion import (
    HIERARCHY_SECTION_TITLE,
    HierarchyTarget,
    expand_with_hierarchy,
    hierarchy_targets,
    select_hierarchy_results,
)

__all__ = [
    "RELATION_COMPATIBLE",
    "RELATION_DEVELOPS",
    "Catalog",
    "CatalogDocument",
    "CatalogRelation",
    "HIERARCHY_SECTION_TITLE",
    "HierarchyTarget",
    "catalog_from_dict",
    "expand_with_hierarchy",
    "hierarchy_targets",
    "load_catalog",
    "resolve_relations",
    "select_hierarchy_results",
]

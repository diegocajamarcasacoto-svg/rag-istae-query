"""Variante de contexto largo: corpus completo como contexto del LLM.

No depende de Qdrant, de modelos de embeddings ni de rerankers.
"""

from query.long_context.catalog import build_catalog_from_matriz, load_catalog
from query.long_context.citations import Citation, parse_answer
from query.long_context.corpus import build_report, load_corpus, render_corpus, units_records
from query.long_context.llm import ChatMessage, LLMClient, LLMRequest, LLMResponse
from query.long_context.models import DocumentInfo, ParsedDocument, Relation, Unit
from query.long_context.variant import Answer, LongContextVariant

__all__ = [
    "Answer",
    "ChatMessage",
    "Citation",
    "DocumentInfo",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "LongContextVariant",
    "ParsedDocument",
    "Relation",
    "Unit",
    "parse_answer",
    "build_catalog_from_matriz",
    "build_report",
    "load_catalog",
    "load_corpus",
    "render_corpus",
    "units_records",
]

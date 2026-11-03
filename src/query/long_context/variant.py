"""Variantes A y A+J: responder con el corpus completo como contexto.

``LongContextVariant`` se construye una vez (al arrancar la API o el script de
evaluación) y luego atiende preguntas con ``answer()``. No guarda estado entre
preguntas: el historial de la conversación lo aporta quien llama.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from query.long_context.citations import Citation, parse_answer
from query.long_context.corpus import CorpusDocument, Variant, render_corpus
from query.long_context.llm import ChatMessage, LLMClient, LLMRequest
from query.long_context.models import Relation

VARIANT_NAMES = {"A": "LLM de contexto largo", "AJ": "LLM de contexto largo + jerarquía"}


@dataclass
class Answer:
    """Respuesta lista para la API, la interfaz de chat y la evaluación."""

    variant: str
    variant_name: str
    question: str
    text: str
    body: str
    citations: list[Citation]
    abstained: bool
    unknown_citations: list[str]
    cited_without_source: list[str]
    verified_quote_ratio: float | None
    corpus_version: str
    provider: str
    model: str
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    latency_ms: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citations"] = [citation.to_dict() for citation in self.citations]
        return data


def load_system_prompt(prompts_dir: Path, variant: Variant) -> str:
    """Prompt base común a todas las variantes, más la capa jerárquica en A+J."""
    prompts_dir = Path(prompts_dir)
    parts = [(prompts_dir / "base_es.txt").read_text(encoding="utf-8").strip()]
    if variant == "AJ":
        parts.append((prompts_dir / "jerarquia_es.txt").read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


class LongContextVariant:
    def __init__(
        self,
        documents: Sequence[CorpusDocument],
        variant: Variant,
        client: LLMClient,
        prompts_dir: Path,
        relations: Sequence[Relation] = (),
        rank_labels: dict[str, str] | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        max_history_messages: int = 6,
    ) -> None:
        if variant not in VARIANT_NAMES:
            raise ValueError(f"Variante no válida: {variant!r}. Use 'A' o 'AJ'.")
        self.variant: Variant = variant
        self.client = client
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_history_messages = max_history_messages
        self.system = load_system_prompt(prompts_dir, variant)
        self.context = render_corpus(list(documents), variant, list(relations), rank_labels or {})
        digest = hashlib.sha256()
        for part in (variant, self.system, self.context, client.model):
            digest.update(part.encode("utf-8"))
        # Identifica exactamente qué instrucciones, corpus y modelo produjeron cada respuesta.
        self.corpus_version = digest.hexdigest()[:16]
        self.units: dict[str, tuple[str, str]] = {
            unit.unit_id: (doc.info.document_id, unit.text)
            for doc in documents
            for unit in doc.parsed.units
        }

    def build_request(self, question: str, history: Sequence[ChatMessage] = ()) -> LLMRequest:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("La pregunta no puede estar vacía.")
        recent = tuple(history)[-self.max_history_messages:] if self.max_history_messages else ()
        return LLMRequest(
            system=self.system,
            context=self.context,
            question=question.strip(),
            history=recent,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            cache_key=self.corpus_version,
        )

    def count_tokens(self) -> int:
        """Tokens del prefijo fijo (instrucciones + corpus) con una pregunta mínima."""
        return self.client.count_tokens(self.build_request("."))

    def _to_answer(self, question: str, text: str, response_meta: dict[str, Any], latency_ms: float) -> Answer:
        parsed = parse_answer(text, self.units)
        return Answer(
            variant=self.variant,
            variant_name=VARIANT_NAMES[self.variant],
            question=question,
            text=text,
            body=parsed.body,
            citations=parsed.citations,
            abstained=parsed.abstained,
            unknown_citations=parsed.unknown_ids,
            cited_without_source=parsed.cited_without_source,
            verified_quote_ratio=parsed.verified_quote_ratio,
            corpus_version=self.corpus_version,
            provider=self.client.name,
            model=response_meta.get("model") or self.client.model,
            input_tokens=response_meta.get("input_tokens"),
            cached_input_tokens=response_meta.get("cached_input_tokens"),
            output_tokens=response_meta.get("output_tokens"),
            finish_reason=response_meta.get("finish_reason"),
            latency_ms=latency_ms,
            extra=response_meta.get("extra", {}),
        )

    def answer(self, question: str, history: Sequence[ChatMessage] = ()) -> Answer:
        request = self.build_request(question, history)
        started = time.perf_counter()
        response = self.client.generate(request)
        latency_ms = (time.perf_counter() - started) * 1000
        meta = {
            "model": response.model,
            "input_tokens": response.input_tokens,
            "cached_input_tokens": response.cached_input_tokens,
            "output_tokens": response.output_tokens,
            "finish_reason": response.finish_reason,
            "extra": response.extra,
        }
        return self._to_answer(request.question, response.text, meta, latency_ms)

    def stream(self, question: str, history: Sequence[ChatMessage] = ()) -> Iterator[str | Answer]:
        """Entrega fragmentos de texto y, al final, el ``Answer`` con las citas verificadas.

        Pensado para el endpoint de chat: el frontend muestra los fragmentos a
        medida que llegan y, al recibir el ``Answer``, presenta las citas.
        """
        request = self.build_request(question, history)
        started = time.perf_counter()
        pieces: list[str] = []
        for piece in self.client.stream(request):
            pieces.append(piece)
            yield piece
        latency_ms = (time.perf_counter() - started) * 1000
        yield self._to_answer(request.question, "".join(pieces), {}, latency_ms)

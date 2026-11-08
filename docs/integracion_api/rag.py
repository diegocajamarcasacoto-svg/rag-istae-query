"""Arquitecturas RAG (B y C): el contexto lo arma el servicio de recuperación.

Archivo propuesto para ``rag-istae-api/app/architectures/rag.py``.

A diferencia de A/AJ, no usa el corpus completo de la API (``Library``):

- la recuperación la hace ``rag-istae-query`` (``POST /retrieve``), que devuelve
  los artículos pertinentes y el contexto ya armado con el formato de la API;
- las citas se verifican contra **los artículos que se enviaron al modelo**;
- las fuentes de la respuesta llevan el texto de cada artículo, para que la web
  lo muestre sin pedir ``GET /units/{id}``.

Conversación (línea base): una conversación = un tema. La recuperación usa la
primera pregunta del usuario; el historial reciente va al modelo como en A/AJ.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx

from app.architectures.base import Answer
from app.llm.base import ChatMessage, LLMClient, LLMRequest, LLMResult
from app.normativa.citations import parse_answer

RagVariant = Literal["B", "C"]


class RetrievalError(RuntimeError):
    """El servicio de recuperación no respondió o respondió con error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RetrievalClient:
    """Cliente de ``POST /retrieve`` (contrato 1.x de rag-istae-query)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 90,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._http = http_client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key},
            timeout=httpx.Timeout(connect=15, read=timeout_seconds, write=30, pool=15),
        )

    def retrieve(self, question: str, *, hierarchy: bool, turn: int) -> dict[str, Any]:
        try:
            response = self._http.post("/retrieve", json={"question": question, "hierarchy": hierarchy, "turn": turn})
        except httpx.TimeoutException as error:
            raise RetrievalError(f"/retrieve: tiempo de espera agotado ({type(error).__name__})", 504) from error
        except httpx.TransportError as error:
            raise RetrievalError(f"/retrieve: error de conexión ({type(error).__name__})", 503) from error
        if response.status_code >= 400:
            raise RetrievalError(f"/retrieve respondió {response.status_code}: {response.text[:200]}", response.status_code)
        return response.json()

    def close(self) -> None:
        self._http.close()


def load_rag_prompts(prompts_dir: Path, variant: RagVariant) -> tuple[str, str]:
    """B = base + rag; C = base + rag + jerarquía para RAG. Recordatorio aparte."""
    prompts_dir = Path(prompts_dir)
    names = ["base_es.txt", "rag_es.txt"] + (["jerarquia_rag_es.txt"] if variant == "C" else [])
    system = "\n\n".join((prompts_dir / name).read_text(encoding="utf-8").strip() for name in names)
    reminder = (prompts_dir / "recordatorio_es.txt").read_text(encoding="utf-8").strip()
    return system, reminder


def conversation_anchor(question: str, history: Sequence[ChatMessage]) -> tuple[str, int]:
    """Consulta de recuperación (primera pregunta del usuario) y número de turno."""
    user_turns = [message.content for message in history if message.role == "user"]
    return (user_turns[0] if user_turns else question).strip(), len(user_turns) + 1


def sources_with_text(citations: list, retrieved: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    """Como ``sources_by_document``, pero cada unidad lleva el texto que vio el modelo."""
    groups: dict[str, dict[str, object]] = {}
    for citation in citations:
        unit = retrieved.get(citation.unit_id)
        if not citation.known or unit is None:
            continue
        group = groups.setdefault(unit["document_id"], {
            "document_id": unit["document_id"],
            "document_name": unit.get("document_name") or unit["document_id"],
            "units": [],
        })
        quote = citation.quote.strip().strip("«»“”\"").strip() if citation.quote else None
        group["units"].append({
            "unit_id": citation.unit_id,
            "quote": quote or None,
            "quote_verified": citation.quote_verified,
            # Nuevos (la web los usa para mostrar el artículo sin GET /units):
            "label": unit.get("label") or "",
            "title": unit.get("title") or "",
            "text": unit.get("text") or "",
            "partial": bool(unit.get("partial")),
            "origin": unit.get("origin") or "base",
        })
    return list(groups.values())


class RagAssistant:
    def __init__(
        self,
        variant: RagVariant,
        retrieval: RetrievalClient,
        client: LLMClient,
        prompts_dir: Path,
        max_history_messages: int = 6,
        max_output_tokens: int = 8192,
    ) -> None:
        self.variant: RagVariant = variant
        self.retrieval = retrieval
        self.client = client
        self.max_history_messages = max_history_messages
        self.max_output_tokens = max_output_tokens
        self.system, self.reminder = load_rag_prompts(prompts_dir, variant)

    def stream(self, question: str, history: Sequence[ChatMessage] = ()) -> Iterator[str | Answer]:
        started = time.perf_counter()
        anchor, turn = conversation_anchor(question, history)
        data = self.retrieval.retrieve(anchor, hierarchy=self.variant == "C", turn=turn)

        recent = tuple(history)[-self.max_history_messages:] if self.max_history_messages else ()
        request = LLMRequest(
            system=self.system,
            context=data["context"],  # ya viene en el formato de la API (=== [DOC] … / [unit_id] …)
            question=question.strip(),
            history=recent,
            reminder=self.reminder,
            max_output_tokens=self.max_output_tokens,
        )
        pieces: list[str] = []
        result = LLMResult(model=self.client.model)
        for item in self.client.stream(request):
            if isinstance(item, LLMResult):
                result = item
            else:
                pieces.append(item)
                yield item
        yield self._answer("".join(pieces), result, data, (time.perf_counter() - started) * 1000)

    def _answer(self, text: str, result: LLMResult, data: dict[str, Any], latency_ms: float) -> Answer:
        retrieved = {unit["unit_id"]: unit for unit in data["units"]}
        # Verificación contra lo que se envió: una cita a un artículo que no estaba en el contexto queda como desconocida.
        unit_texts = {unit_id: (unit["document_id"], unit["text"]) for unit_id, unit in retrieved.items()}
        parsed = parse_answer(text, unit_texts)
        return Answer(
            body=parsed.body,
            sources=sources_with_text(parsed.citations, retrieved),
            abstained=parsed.abstained,
            unknown_citations=parsed.unknown_ids,
            cited_without_source=parsed.cited_without_source,
            external_sources=parsed.external_sources,
            verified_quote_ratio=parsed.verified_quote_ratio,
            architecture=self.variant,
            corpus_version=self._corpus_version(data, result.model),
            model=result.model,
            input_tokens=result.input_tokens,
            cached_input_tokens=result.cached_input_tokens,
            output_tokens=result.output_tokens,
            reasoning_tokens=result.reasoning_tokens,
            finish_reason=result.finish_reason,
            latency_ms=round(latency_ms),
        )

    def _corpus_version(self, data: dict[str, Any], model: str) -> str:
        """Identifica instrucciones, índice de recuperación, línea base y modelo que produjeron la respuesta."""
        index = data.get("index_version") or {}
        digest = hashlib.sha256()
        for part in (self.variant, self.system, self.reminder, data.get("baseline", ""),
                     index.get("units_sha256", ""), index.get("catalog_sha256", ""), model):
            digest.update(str(part).encode("utf-8"))
        return digest.hexdigest()[:16]

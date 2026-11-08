"""Aplicación FastAPI del servicio de recuperación.

    uvicorn query.service.app:create_app --factory      (o scripts/serve_retrieval.py)

- ``GET /health``: estado, línea base y versión del índice (sin secretos).
- ``POST /retrieve``: artículos + contexto renderizado para B (``hierarchy=false``)
  o C (``hierarchy=true``). No genera respuesta: la API arma el prompt y llama al LLM.

Conversación (línea base): una conversación = un tema. La API envía siempre la
primera pregunta del usuario, así que los turnos de seguimiento obtienen el mismo
contexto; el caché evita repetir la recuperación.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from query.engine import RetrievalEngine, RetrievalOutput
from query.run_log import group_timings, new_run_id, write_run
from query.service.schemas import CONTRACT_VERSION, Mode, RetrieveRequest, RetrieveResponse
from query.service.settings import ServiceSettings, load_settings

LOGGER = logging.getLogger("query.service")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

EngineFactory = Callable[[ServiceSettings], Any]


def _default_engine_factory(settings: ServiceSettings) -> RetrievalEngine:
    return RetrievalEngine.from_config(settings.retrieval_config)


class RetrievalCache:
    """LRU en memoria: misma consulta y modo → mismo resultado (la recuperación es determinista)."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[tuple, RetrievalOutput] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(request: RetrieveRequest) -> tuple:
        return (request.question, request.rerank, request.hierarchy, tuple(sorted(request.document_ids or ())))

    def get(self, key: tuple) -> RetrievalOutput | None:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key: tuple, value: RetrievalOutput) -> None:
        if self.size <= 0:
            return
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)


def _ms(seconds: Any) -> int | None:
    return round(seconds * 1000) if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else None


def create_app(
    settings: ServiceSettings | None = None,
    engine_factory: EngineFactory | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    engine_factory = engine_factory or _default_engine_factory
    cache = RetrievalCache(settings.cache_size)
    status: dict[str, Any] = {"engine": None, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            status["engine"] = engine_factory(settings)
        except Exception as exc:  # noqa: BLE001 - /health informa el error; /retrieve responde 503
            LOGGER.exception("No se pudo cargar el motor de recuperación.")
            status["error"] = f"{type(exc).__name__}: {exc}"
        yield
        engine = status["engine"]
        if engine is not None and hasattr(engine, "close"):
            engine.close()

    app = FastAPI(
        title="RAG ISTAE — servicio de recuperación",
        version=CONTRACT_VERSION,
        description="Recuperación de artículos para las arquitecturas B (RAG híbrido) y C (RAG jerárquico).",
        lifespan=lifespan,
    )

    def require_api_key(api_key: str | None = Depends(API_KEY_HEADER)) -> None:
        if settings.auth_disabled:
            return
        if not api_key or not secrets.compare_digest(api_key, settings.api_key or ""):
            raise HTTPException(status_code=401, detail="Falta X-API-Key o no es válida.")

    def engine_or_503() -> Any:
        if status["engine"] is None:
            raise HTTPException(status_code=503, detail="El motor de recuperación no está disponible.")
        return status["engine"]

    def log_run(record: dict[str, Any]) -> None:
        if settings.log_dir is None:
            return
        try:
            write_run(settings.log_dir, record)
        except Exception:  # noqa: BLE001 - el registro no debe tumbar la respuesta
            LOGGER.exception("No se pudo registrar la ejecución %s.", record.get("run_id"))

    def base_record(run_id: str, body: RetrieveRequest, engine: Any) -> dict[str, Any]:
        params = engine.params
        return {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": "service",
            "tag": body.tag,
            "question_id": body.question_id,
            "query": body.question,
            "args": {
                "rerank": body.rerank,
                "top_k": params.top_k,
                "candidate_k": params.candidate_k,
                "prefetch_k": params.prefetch_k,
                "rrf_k": params.rrf_k,
                "expansion": "units",
                "retrieval_mode": "hierarchical" if body.hierarchy else "hybrid",
                "max_context_tokens": params.max_context_tokens,
                "document_ids": body.document_ids,
                "turn": body.turn,
            },
            "baseline": params.to_dict(),
            "index_version": engine.index_version,
            "service": settings.describe(),
        }

    @app.get("/health")
    def health() -> JSONResponse:
        engine = status["engine"]
        if engine is None:
            return JSONResponse(
                status_code=503,
                content={"status": "error" if status["error"] else "loading", "error": status["error"]},
            )
        public = {key: value for key, value in settings.describe().items() if key != "log_dir"}
        return JSONResponse({"status": "ok", "contract_version": CONTRACT_VERSION, **engine.info(),
                             "service": public})

    @app.post("/retrieve", response_model=RetrieveResponse, response_model_exclude_none=True,
              dependencies=[Depends(require_api_key)])
    def retrieve(body: RetrieveRequest, request: Request) -> RetrieveResponse:
        engine = engine_or_503()
        unknown = sorted(set(body.document_ids or ()) - set(engine.document_ids))
        if unknown:
            raise HTTPException(status_code=422, detail=f"document_ids desconocidos: {', '.join(unknown)}")

        run_id = new_run_id()
        record = base_record(run_id, body, engine)
        started = time.perf_counter()
        key = RetrievalCache.key(body)
        output = cache.get(key)
        cached = output is not None
        try:
            if output is None:
                output = engine.retrieve(
                    body.question, rerank=body.rerank, hierarchy=body.hierarchy, document_ids=body.document_ids,
                )
                cache.put(key, output)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Falla en /retrieve (%s).", run_id)
            log_run({**record, "status": "error", "cached": False,
                     "timings_seconds": {"total": time.perf_counter() - started},
                     "error": {"stage": "retrieval", "type": type(exc).__name__, "message": str(exc)}})
            raise HTTPException(status_code=500, detail=f"Error interno en la recuperación (run_id {run_id}).")

        total = time.perf_counter() - started
        timings = {**output.timings, "total": total} if not cached else {"total": total}
        log_run({
            **record,
            "status": "ok",
            "cached": cached,
            "timings_seconds": timings,
            "context_result": output.context_result.to_dict() if output.context_result else None,
            "retrieval_results": output.debug.get("candidates"),
        })

        grouped = group_timings(output.timings)
        debug = None
        if body.debug:
            debug = {
                **output.debug,
                "timings_ms": {
                    "execution": {stage: _ms(value) for stage, value in grouped["execution"].items()},
                    "nested": {stage: _ms(value) for stage, value in grouped["nested"].items()},
                    "execution_total": _ms(grouped["execution_total"]),
                    "engine_load": {stage: _ms(value) for stage, value in engine.load_timings.items()},
                    "from_cache": cached,
                },
                "baseline_params": engine.params.to_dict(),
            }
        return RetrieveResponse(
            run_id=run_id,
            baseline=engine.params.id,
            mode=Mode(rerank=body.rerank, hierarchy=body.hierarchy, architecture="C" if body.hierarchy else "B"),
            units=output.units,
            context=output.context,
            tokens_estimated=output.tokens_estimated,
            warnings=output.warnings,
            cached=cached,
            index_version=engine.index_version,
            timings_ms={"total": _ms(total)},
            debug=debug,
        )

    return app

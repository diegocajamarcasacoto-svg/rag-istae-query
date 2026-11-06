"""Registro automático de ejecuciones del pipeline RAG para evaluación posterior.

Cada ejecución produce:

- ``runs/<run_id>.json``: detalle completo (respuesta, contexto, rankings,
  tiempos, parámetros y configuración sin secretos).
- ``runs_index.jsonl``: una línea resumen por ejecución.
- ``runs_index.csv``: el mismo resumen con separador ``;`` y UTF-8 con BOM,
  para abrirlo directamente en Excel con configuración regional en español.

En el índice los tiempos se guardan en milisegundos enteros para evitar
ambigüedades con el separador decimal.

Si las columnas del índice cambian entre versiones, el CSV se regenera desde
``runs_index.jsonl`` (fuente de verdad) con el encabezado nuevo; las columnas
que no existían quedan vacías en las ejecuciones anteriores.
"""

from __future__ import annotations

import csv
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

INDEX_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "status",
    "error_stage",
    "error",
    "tag",
    "question_id",
    "query",
    "provider",
    "model",
    "extra_body",
    "temperature",
    "max_output_tokens",
    "rerank",
    "top_k",
    "candidate_k",
    "prefetch_k",
    "rrf_k",
    "expansion",
    "seed_k",
    "neighbors_before",
    "neighbors_after",
    "max_context_tokens",
    "n_candidates",
    "n_context_chunks",
    "context_chunk_ids",
    "context_tokens",
    "input_tokens",
    "output_tokens",
    "finish_reason",
    "generation_ms",
    "total_ms",
    "answer",
    "detail_file",
)

CSV_DELIMITER = ";"
RUNS_DIRNAME = "runs"
INDEX_JSONL = "runs_index.jsonl"
INDEX_CSV = "runs_index.csv"


def new_run_id(now: datetime | None = None) -> str:
    """Identificador ordenable por fecha: ``YYYYMMDD-HHMMSS_<6 hex>``."""
    moment = now or datetime.now()
    return f"{moment:%Y%m%d-%H%M%S}_{secrets.token_hex(3)}"


def _ms(seconds: Any) -> int | None:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return None
    return round(seconds * 1000)


def build_index_row(record: dict[str, Any]) -> dict[str, Any]:
    """Aplana un registro detallado a las columnas de ``INDEX_COLUMNS``.

    Las claves ausentes (ejecuciones fallidas, ``tag`` o ``question_id`` no
    indicados) quedan como ``None``.
    """
    args = record.get("args") or {}
    generation_config = record.get("generation_config") or {}
    generation = record.get("generation_result") or {}
    metrics = generation.get("provider_metrics") or {}
    context = generation.get("context_result") or {}
    chunks = context.get("chunks") or []
    timings = record.get("timings_seconds") or {}
    error = record.get("error") or {}
    retrieval = record.get("retrieval_results")
    extra_body = generation_config.get("extra_body")

    error_text = None
    if error:
        error_text = f"{error.get('type')}: {error.get('message')}"

    row = {
        "run_id": record.get("run_id"),
        "timestamp": record.get("timestamp"),
        "status": record.get("status"),
        "error_stage": error.get("stage"),
        "error": error_text,
        "tag": record.get("tag"),
        "question_id": record.get("question_id"),
        "query": record.get("query"),
        "provider": generation.get("provider") or generation_config.get("provider"),
        "model": generation.get("model") or generation_config.get("model"),
        "extra_body": (
            json.dumps(extra_body, ensure_ascii=False, sort_keys=True)
            if extra_body else None
        ),
        "temperature": generation_config.get("temperature"),
        "max_output_tokens": generation_config.get("max_output_tokens"),
        "rerank": args.get("rerank"),
        "top_k": args.get("top_k"),
        "candidate_k": args.get("candidate_k"),
        "prefetch_k": args.get("prefetch_k"),
        "rrf_k": args.get("rrf_k"),
        "expansion": (args.get("expansion") or "neighbors") if args else None,
        "seed_k": args.get("seed_k"),
        "neighbors_before": args.get("neighbors_before"),
        "neighbors_after": args.get("neighbors_after"),
        "max_context_tokens": args.get("max_context_tokens"),
        "n_candidates": len(retrieval) if isinstance(retrieval, list) else None,
        "n_context_chunks": len(chunks) if context else None,
        "context_chunk_ids": (
            "|".join(str(chunk.get("chunk_id")) for chunk in chunks) if chunks else None
        ),
        "context_tokens": ((context.get("metadata") or {}).get("metrics") or {}).get("token_count"),
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "finish_reason": generation.get("finish_reason"),
        "generation_ms": _ms(timings.get("generation")),
        "total_ms": _ms(timings.get("total")),
        "answer": generation.get("answer"),
        "detail_file": record.get("detail_file"),
    }
    return {column: row.get(column) for column in INDEX_COLUMNS}


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_run(log_dir: Path, record: dict[str, Any]) -> Path:
    """Guarda el detalle y agrega la fila resumen al JSONL y al CSV.

    ``record`` debe incluir ``run_id``. Se añade ``detail_file`` (ruta relativa
    a ``log_dir``) antes de escribir. Devuelve la ruta del JSON de detalle.
    """
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("El registro debe incluir run_id.")

    log_dir = Path(log_dir)
    runs_dir = log_dir / RUNS_DIRNAME
    runs_dir.mkdir(parents=True, exist_ok=True)
    detail_path = runs_dir / f"{run_id}.json"
    record = {**record, "detail_file": f"{RUNS_DIRNAME}/{run_id}.json"}

    detail_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    row = build_index_row(record)
    with (log_dir / INDEX_JSONL).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    csv_path = log_dir / INDEX_CSV
    if _csv_header_matches(csv_path):
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS, delimiter=CSV_DELIMITER)
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    else:
        # CSV nuevo o con columnas de una versión anterior: se regenera desde el JSONL,
        # que ya incluye esta ejecución.
        _rewrite_csv(csv_path, log_dir / INDEX_JSONL)

    return detail_path


def _csv_header_matches(csv_path: Path) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter=CSV_DELIMITER), [])
    return tuple(header) == INDEX_COLUMNS


def _rewrite_csv(csv_path: Path, jsonl_path: Path) -> None:
    rows = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS, delimiter=CSV_DELIMITER)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in INDEX_COLUMNS})

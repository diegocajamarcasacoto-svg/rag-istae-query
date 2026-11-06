import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from query.context.models import ContextChunk, ContextResult
from query.generation.errors import GenerationProviderError
from query.generation.generation_config import GenerationConfig
from query.generation.models import GenerationResult, ProviderMetrics
from query.run_log import INDEX_COLUMNS, build_index_row, new_run_id, write_run

import query_rag

SECRET = "sk-secreto-no-debe-aparecer"


def _record(run_id: str, **overrides) -> dict:
    record = {
        "run_id": run_id,
        "timestamp": "2026-11-06T15:30:12-05:00",
        "status": "ok",
        "error": None,
        "tag": "nemotron",
        "question_id": "P01",
        "query": "¿Qué dispone el artículo 18?",
        "generation_result": {
            "answer": 'Línea 1; con "comillas"\nLínea 2',
            "provider": "openrouter",
            "model": "nvidia/nemotron",
            "provider_metrics": {"input_tokens": 100, "output_tokens": 20},
            "finish_reason": "stop",
            "context_result": {"chunks": [{"chunk_id": "INT08:00055"}, {"chunk_id": "INT08:00056"}]},
        },
        "retrieval_results": [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}],
        "ranking_results": [],
        "timings_seconds": {"generation": 1.2346, "total": 9.8765},
        "args": {"rerank": True, "top_k": 5, "seed_k": 3},
        "generation_config": {
            "provider": "openrouter",
            "model": "nvidia/nemotron",
            "temperature": 0.2,
            "max_output_tokens": 2048,
            "extra_body": {"reasoning": {"effort": "none"}},
        },
    }
    record.update(overrides)
    return record


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


class RunLogTests(unittest.TestCase):
    def test_run_id_is_sortable_by_date(self) -> None:
        run_id = new_run_id(datetime(2026, 11, 6, 15, 30, 12))
        self.assertRegex(run_id, r"^20261106-153012_[0-9a-f]{6}$")

    def test_write_run_creates_detail_jsonl_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "logs"
            first = write_run(log_dir, _record("r1"))
            write_run(log_dir, _record("r2"))

            self.assertEqual(first, log_dir / "runs" / "r1.json")
            detail = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(detail["detail_file"], "runs/r1.json")
            self.assertEqual(detail["generation_result"]["model"], "nvidia/nemotron")

            lines = (log_dir / "runs_index.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["run_id"] for line in lines], ["r1", "r2"])

            raw = (log_dir / "runs_index.csv").read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
            self.assertEqual(raw.count(b"\xef\xbb\xbf"), 1)
            rows = _read_csv(log_dir / "runs_index.csv")

        self.assertEqual(len(rows), 2)
        self.assertEqual(tuple(rows[0]), INDEX_COLUMNS)
        row = rows[0]
        self.assertEqual(row["answer"], 'Línea 1; con "comillas"\nLínea 2')
        self.assertEqual(row["context_chunk_ids"], "INT08:00055|INT08:00056")
        self.assertEqual(row["n_candidates"], "3")
        self.assertEqual(row["n_context_chunks"], "2")
        self.assertEqual(row["generation_ms"], "1235")
        self.assertEqual(row["total_ms"], "9876")
        self.assertEqual(row["rerank"], "true")
        self.assertEqual(json.loads(row["extra_body"]), {"reasoning": {"effort": "none"}})

    def test_tag_and_question_id_are_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            write_run(log_dir, _record("r1", tag=None, question_id=None))
            row = _read_csv(log_dir / "runs_index.csv")[0]
            line = json.loads((log_dir / "runs_index.jsonl").read_text(encoding="utf-8"))
        self.assertEqual((row["tag"], row["question_id"]), ("", ""))
        self.assertIsNone(line["tag"])
        self.assertIsNone(line["question_id"])

    def test_error_record_has_empty_generation_columns(self) -> None:
        record = _record(
            "r1",
            status="error",
            error={"type": "GenerationProviderError", "message": "HTTP 429", "stage": "generation"},
            generation_result=None,
        )
        row = build_index_row(record)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error_stage"], "generation")
        self.assertEqual(row["error"], "GenerationProviderError: HTTP 429")
        self.assertEqual(row["provider"], "openrouter")
        self.assertIsNone(row["answer"])
        self.assertIsNone(row["input_tokens"])
        self.assertIsNone(row["n_context_chunks"])

    def test_missing_run_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_run(Path(directory), _record(""))


def _point(chunk_id: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(payload={"chunk_id": chunk_id}, score=score)


def _generation_config() -> GenerationConfig:
    return GenerationConfig(
        provider="openrouter",
        model="nvidia/nemotron",
        system_prompt_path=Path("system.txt"),
        temperature=0.2,
        max_output_tokens=256,
        timeout_seconds=30,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        extra_headers={"X-Title": "RAG ISTAE", "Authorization": f"Bearer {SECRET}"},
        extra_body={},
    )


class QueryRagLoggingTests(unittest.TestCase):
    """Ejecuta ``query_rag.main`` con el pipeline simulado (sin modelos ni red)."""

    def _run(self, argv: list[str], *, generate_side_effect=None):
        context_result = ContextResult(
            query="q",
            context="ctx",
            chunks=[ContextChunk(chunk_id="INT08:00055", document_id="INT08", chunk_index=55, source_file="n.pdf", page_numbers=[], headings=[], item_types=[], text="t")],
            metadata={},
        )

        def _generate(context, config):
            if generate_side_effect is not None:
                raise generate_side_effect
            return GenerationResult(
                answer="Respuesta",
                provider="openrouter",
                model="nvidia/nemotron",
                provider_metrics=ProviderMetrics(input_tokens=10, output_tokens=5),
                application_latency_ms=1.0,
                finish_reason="stop",
                context_result=context,
            )

        local_config = {"collection_name": "istae_chunks_e5_base", "reranker_model_path": "bge"}
        patches = [
            mock.patch.object(query_rag, "load_config", return_value=local_config),
            mock.patch.object(query_rag, "load_generation_config", return_value=_generation_config()),
            mock.patch.object(query_rag, "load_e5", return_value=object()),
            mock.patch.object(query_rag, "open_client", return_value=mock.MagicMock()),
            mock.patch.object(query_rag, "encode_query", return_value=[0.0]),
            mock.patch.object(
                query_rag, "hybrid_search", return_value=[_point("INT08:00055", 0.5), _point("X:1", 0.4)]
            ),
            mock.patch.object(query_rag, "build_context", return_value=context_result),
            mock.patch.object(query_rag, "generate_answer", side_effect=_generate),
            mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": SECRET}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return query_rag.main(argv)

    def test_successful_run_is_logged_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            self._run(["pregunta", "--no-rerank", "--log-dir", str(log_dir)])
            details = list((log_dir / "runs").glob("*.json"))
            text = details[0].read_text(encoding="utf-8")
            row = _read_csv(log_dir / "runs_index.csv")[0]

        self.assertEqual(len(details), 1)
        self.assertNotIn(SECRET, text)
        detail = json.loads(text)
        self.assertEqual(detail["status"], "ok")
        self.assertIsNone(detail["question_id"])
        self.assertEqual(detail["generation_config"]["api_key_env"], "OPENROUTER_API_KEY")
        self.assertNotIn("Authorization", detail["generation_config"]["extra_headers"])
        self.assertEqual(row["answer"], "Respuesta")
        self.assertEqual(row["context_chunk_ids"], "INT08:00055")
        self.assertEqual(row["rerank"], "false")

    def test_failed_generation_is_logged_and_reraised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            with self.assertRaises(GenerationProviderError):
                self._run(
                    ["pregunta", "--no-rerank", "--log-dir", str(log_dir), "--question-id", "P07"],
                    generate_side_effect=GenerationProviderError("HTTP 429"),
                )
            row = _read_csv(log_dir / "runs_index.csv")[0]

        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error_stage"], "generation")
        self.assertEqual(row["question_id"], "P07")
        self.assertEqual(row["n_candidates"], "2")
        self.assertEqual(row["answer"], "")

    def test_no_log_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "logs"
            self._run(["pregunta", "--no-rerank", "--no-log", "--log-dir", str(log_dir)])
            self.assertFalse(log_dir.exists())


if __name__ == "__main__":
    unittest.main()

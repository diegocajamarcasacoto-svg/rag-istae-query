"""Pruebas de la expansión por artículo (query.units) y su integración en query_rag."""

import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from query.generation.generation_config import GenerationConfig
from query.generation.models import GenerationResult, ProviderMetrics
from query.retrieval import hybrid_search
from query.run_log import INDEX_COLUMNS, write_run
from query.units import PARTIAL_SEPARATOR, Unit, UnitIndex, build_unit_context, estimate_tokens, load_units

import query_rag
from query import pipeline


def _point(chunk_id: str, score: float = 0.5, **payload) -> SimpleNamespace:
    document_id = chunk_id.split(":")[0]
    return SimpleNamespace(
        payload={"chunk_id": chunk_id, "document_id": document_id, "source_file": f"{document_id}.pdf",
                 "page_numbers": [1], "text": f"texto {chunk_id}", **payload},
        score=score,
    )


def _index() -> UnitIndex:
    return UnitIndex([
        Unit("INT08-art11", "INT08", "articulo", "Art. 11", "Declaratoria de Aptitud", "",
             ("INT08:00037", "INT08:00038"), "Artículo 11. Declaratoria…\n\n- a) Certificados."),
        Unit("INT08-art12", "INT08", "articulo", "Art. 12", "Declaración de egresado", "",
             ("INT08:00039", "INT08:00040", "INT08:00041"), "Artículo 12. Egresado…\n\n- a)…\n\nEl estudiante…"),
        Unit("EXT06-disp-transitoria-tercera", "EXT06", "disposicion", "Disposición transitoria Tercera", "", "",
             tuple(f"EXT06:{i:05d}" for i in range(1, 11)), "x" * 20000),
    ])


class UnitIndexTests(unittest.TestCase):
    def test_maps_chunks_to_units(self) -> None:
        index = _index()
        self.assertEqual(index.unit_for_chunk("INT08:00040").unit_id, "INT08-art12")
        self.assertIsNone(index.unit_for_chunk("INT99:00001"))
        self.assertEqual(len(index), 3)
        self.assertEqual(index.chunk_count, 15)

    def test_rejects_chunk_in_two_units(self) -> None:
        with self.assertRaises(ValueError):
            UnitIndex([
                Unit("A-art1", "A", "articulo", "", "", "", ("A:00001",), "a"),
                Unit("A-art2", "A", "articulo", "", "", "", ("A:00001",), "b"),
            ])

    def test_load_units_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "units.jsonl"
            path.write_text(
                json.dumps({"unit_id": "INT01-art1", "document_id": "INT01", "kind": "articulo",
                            "label": "Art. 1", "chunk_ids": ["INT01:00001"], "text": "Artículo 1."}) + "\n",
                encoding="utf-8",
            )
            index = load_units(path)
            with self.assertRaises(FileNotFoundError):
                load_units(Path(directory) / "no-existe.jsonl")
        self.assertEqual(index.unit_for_chunk("INT01:00001").label, "Art. 1")


class BuildUnitContextTests(unittest.TestCase):
    def test_deduplicates_units_and_keeps_rank_order(self) -> None:
        ranked = [
            (_point("INT08:00040"), 0.9),
            (_point("INT08:00038"), 0.8),
            (_point("INT08:00039"), 0.7),
        ]
        result = build_unit_context("q", ranked, _index(), max_tokens=None)
        self.assertEqual([c.chunk_id for c in result.chunks], ["INT08-art12", "INT08-art11"])
        art12 = result.chunks[0]
        self.assertEqual(art12.included_by_seed_ids, ["INT08:00040", "INT08:00039"])
        self.assertEqual(art12.reranker_score, 0.9)
        self.assertEqual(art12.inclusion_reason, ["unit_full"])
        self.assertTrue(result.context.startswith("[INT08-art12] INT08 · Art. 12 · Declaración de egresado\n"))
        self.assertIn("El estudiante…", result.context)
        self.assertEqual(result.metadata["expansion"], "units")
        self.assertEqual(result.metadata["metrics"]["token_count"], estimate_tokens(result.context))

    def test_long_unit_sends_only_matched_spans_within_unit(self) -> None:
        fetched_ids = []

        def fetch(chunk_ids):
            fetched_ids.extend(chunk_ids)
            return {chunk_id: f"parte {chunk_id}" for chunk_id in chunk_ids}

        ranked = [(_point("EXT06:00002"), 0.9), (_point("EXT06:00008"), 0.8)]
        result = build_unit_context("q", ranked, _index(), fetch_chunk_texts=fetch,
                                    max_tokens=None, max_unit_tokens=1500, partial_window=1)
        self.assertEqual(fetched_ids, ["EXT06:00001", "EXT06:00002", "EXT06:00003",
                                       "EXT06:00007", "EXT06:00008", "EXT06:00009"])
        chunk = result.chunks[0]
        self.assertEqual(chunk.inclusion_reason, ["unit_partial"])
        self.assertIn(PARTIAL_SEPARATOR, chunk.text)
        self.assertNotIn("xxxx", chunk.text)
        self.assertEqual(result.metadata["metrics"]["units_partial"], 1)

    def test_long_unit_without_fetcher_is_sent_whole_with_warning(self) -> None:
        result = build_unit_context("q", [(_point("EXT06:00002"), 0.9)], _index(), max_tokens=None)
        self.assertEqual(result.chunks[0].inclusion_reason, ["unit_full"])
        self.assertTrue(result.metadata["warnings"])

    def test_budget_keeps_first_unit_and_excludes_what_does_not_fit(self) -> None:
        index = _index()
        first_block_tokens = estimate_tokens(
            "[INT08-art12] INT08 · Art. 12 · Declaración de egresado\n" + index.unit("INT08-art12").text
        )
        ranked = [(_point("INT08:00039"), 0.9), (_point("INT08:00037"), 0.8)]
        result = build_unit_context("q", ranked, index, max_tokens=first_block_tokens)
        self.assertEqual([c.chunk_id for c in result.chunks], ["INT08-art12"])
        self.assertEqual(result.metadata["excluded_units"][0]["unit_id"], "INT08-art11")

    def test_chunk_missing_from_index_is_included_alone(self) -> None:
        result = build_unit_context("q", [(_point("INT99:00001"), 0.5)], _index(), max_tokens=None)
        self.assertEqual(result.chunks[0].chunk_id, "INT99:00001")
        self.assertIn("texto INT99:00001", result.context)
        self.assertTrue(result.metadata["warnings"])

    def test_rrf_scores_are_used_when_reranker_is_off(self) -> None:
        result = build_unit_context("q", [(_point("INT08:00037", score=0.03), None)], _index(), max_tokens=None)
        self.assertAlmostEqual(result.chunks[0].reranker_score, 0.03)


class HybridSearchFilterTests(unittest.TestCase):
    def test_document_ids_filter_is_applied_to_both_prefetches(self) -> None:
        client = mock.MagicMock()
        client.query_points.return_value.points = []
        hybrid_search(client, {"collection_name": "c"}, [0.0], "q", 20, 20, 60, document_ids=["EXT01"])
        kwargs = client.query_points.call_args.kwargs
        self.assertEqual(kwargs["query_filter"].must[0].key, "document_id")
        self.assertEqual(kwargs["query_filter"].must[0].match.any, ["EXT01"])
        self.assertTrue(all(prefetch.filter is not None for prefetch in kwargs["prefetch"]))

    def test_without_document_ids_there_is_no_filter(self) -> None:
        client = mock.MagicMock()
        client.query_points.return_value.points = []
        hybrid_search(client, {"collection_name": "c"}, [0.0], "q", 20, 20, 60)
        kwargs = client.query_points.call_args.kwargs
        self.assertIsNone(kwargs["query_filter"])
        self.assertTrue(all(prefetch.filter is None for prefetch in kwargs["prefetch"]))


class QueryRagUnitsTests(unittest.TestCase):
    def test_units_expansion_end_to_end_with_mocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            units_path = Path(directory) / "units.jsonl"
            units_path.write_text(
                json.dumps({"unit_id": "INT08-art12", "document_id": "INT08", "kind": "articulo",
                            "label": "Art. 12", "title": "Egresado",
                            "chunk_ids": ["INT08:00039", "INT08:00040"], "text": "Artículo 12…"}) + "\n",
                encoding="utf-8",
            )
            log_dir = Path(directory) / "logs"
            captured = {}

            def _generate(context, config):
                captured["context"] = context
                return GenerationResult(
                    answer="Respuesta", provider="openrouter", model="m",
                    provider_metrics=ProviderMetrics(input_tokens=10, output_tokens=5),
                    application_latency_ms=1.0, finish_reason="stop", context_result=context,
                )

            generation_config = GenerationConfig(
                provider="openrouter", model="m", system_prompt_path=Path("s.txt"), temperature=0.2,
                max_output_tokens=256, timeout_seconds=30, api_key_env="OPENROUTER_API_KEY",
                base_url="https://openrouter.ai/api/v1", extra_headers={}, extra_body={},
            )
            patches = [
                mock.patch.object(query_rag, "load_config",
                                  return_value={"collection_name": "c", "reranker_model_path": "bge"}),
                mock.patch.object(query_rag, "load_generation_config", return_value=generation_config),
                mock.patch.object(query_rag, "load_e5", return_value=object()),
                mock.patch.object(query_rag, "open_client", return_value=mock.MagicMock()),
                # La rama units corre en query.pipeline (compartido con el servicio /retrieve).
                mock.patch.object(pipeline, "encode_query", return_value=[0.0]),
                mock.patch.object(pipeline, "hybrid_search",
                                  return_value=[_point("INT08:00040", 0.5), _point("INT08:00039", 0.4)]),
                mock.patch.object(query_rag, "build_context", side_effect=AssertionError("no debe usarse")),
                mock.patch.object(query_rag, "generate_answer", side_effect=_generate),
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "x"}),
            ]
            for patch in patches:
                patch.start()
                self.addCleanup(patch.stop)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                query_rag.main(["pregunta", "--no-rerank", "--expansion", "units",
                                "--units-path", str(units_path), "--log-dir", str(log_dir)])
            with (log_dir / "runs_index.csv").open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle, delimiter=";"))

        context = captured["context"]
        self.assertEqual([c.chunk_id for c in context.chunks], ["INT08-art12"])
        self.assertEqual(row["expansion"], "units")
        self.assertEqual(row["hierarchy_ms"], "")  # sin jerarquía en modo hybrid
        self.assertEqual(row["context_chunk_ids"], "INT08-art12")
        self.assertEqual(row["context_tokens"], str(context.metadata["metrics"]["token_count"]))

    def test_units_expansion_requires_units_path(self) -> None:
        with mock.patch.object(query_rag, "load_config", return_value={"collection_name": "c"}), \
             mock.patch.object(query_rag, "load_generation_config",
                               return_value=SimpleNamespace(api_key_env="K", provider="p", model="m",
                                                            base_url=None, system_prompt_path="s",
                                                            temperature=0, max_output_tokens=1,
                                                            timeout_seconds=1, extra_headers={},
                                                            extra_body={})), \
             mock.patch.object(query_rag, "load_e5", return_value=object()), \
             mock.patch.dict(os.environ, {"K": "x"}), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(ValueError):
                query_rag.main(["pregunta", "--no-rerank", "--expansion", "units", "--no-log"])


class RunLogMigrationTests(unittest.TestCase):
    def test_csv_with_old_header_is_rebuilt_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            old_columns = [c for c in INDEX_COLUMNS if c not in ("expansion", "context_tokens")]
            (log_dir / "runs_index.jsonl").write_text(
                json.dumps({"run_id": "old", "status": "ok", "query": "q"}) + "\n", encoding="utf-8"
            )
            with (log_dir / "runs_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(old_columns)
                writer.writerow(["old", "", "ok"] + [""] * (len(old_columns) - 3))

            write_run(log_dir, {"run_id": "new", "status": "ok", "args": {"expansion": "units"}})
            with (log_dir / "runs_index.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter=";"))

        self.assertEqual(tuple(rows[0]), INDEX_COLUMNS)
        self.assertEqual([row["run_id"] for row in rows], ["old", "new"])
        self.assertEqual(rows[0]["expansion"], "")
        self.assertEqual(rows[1]["expansion"], "units")


if __name__ == "__main__":
    unittest.main()

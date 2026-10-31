from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from query.context import build_context
from query.context.context_merger import deduplicate_and_sort
from query.context.models import ContextChunk
from query.context.neighbor_expander import expand_neighbors
from query.context.structure_detector import detect_structure
from query.context.structure_validator import validate_neighbor
from query.context.token_budget import apply_token_budget
from validate_context_pipeline import _parser, export_validation_json, main


COLLECTION = "test_collection"


def point(chunk_id: str | None, document_id: str | None, index: int | None, text: str, **extra):
    payload = {"text": text, **extra}
    if chunk_id is not None:
        payload["chunk_id"] = chunk_id
    if document_id is not None:
        payload["document_id"] = document_id
    if index is not None:
        payload["chunk_index"] = index
    return SimpleNamespace(id=chunk_id or f"{document_id}:{index}", payload=payload, score=0.5)


def config(**options):
    return {"collection_name": COLLECTION, "context_builder": options}


def empty_client():
    client = MagicMock()
    client.scroll.return_value = ([], None)
    return client


def seed_chunk(
    chunk_id="DOC:00001",
    doc="DOC",
    index=1,
    text="Texto inicial.",
    score=0.9,
    **payload,
):
    return point(chunk_id, doc, index, text, **payload), score


class ContextBuilderTests(unittest.TestCase):
    def test_selects_top_three_seeds_in_reranker_order(self):
        ranked = [
            seed_chunk(f"DOC:{index}", "DOC", index, f"Contenido {index}", 1 - index / 10)
            for index in range(1, 5)
        ]
        result = build_context("consulta", ranked, empty_client(), config())
        self.assertEqual([chunk.chunk_id for chunk in result.chunks], ["DOC:1", "DOC:2", "DOC:3"])
        self.assertEqual(
            [chunk.reranker_score for chunk in result.chunks],
            [0.9, 0.8, 0.7],
        )

    def test_uses_available_results_when_fewer_than_three(self):
        ranked = [seed_chunk("DOC:1", "DOC", 1), seed_chunk("DOC:2", "DOC", 2)]
        result = build_context("consulta", ranked, empty_client(), config())
        self.assertEqual(result.metadata["metrics"]["seed_count"], 2)

    def test_neighbor_query_filters_document_and_index_range(self):
        seed = ContextChunk("DOC:2", "DOC", 2, None, [], [], [], "seed", role="seed")
        neighbor = point("DOC:1", "DOC", 1, "vecino")
        client = MagicMock()
        client.scroll.return_value = ([neighbor], None)
        result = expand_neighbors(client, COLLECTION, seed, before=1, after=1)
        self.assertEqual(result, [neighbor])
        kwargs = client.scroll.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], COLLECTION)
        self.assertTrue(kwargs["with_payload"])
        self.assertFalse(kwargs["with_vectors"])
        conditions = kwargs["scroll_filter"].must
        self.assertEqual(conditions[0].key, "document_id")
        self.assertEqual(conditions[0].match.value, "DOC")
        self.assertEqual(conditions[1].key, "chunk_index")
        self.assertEqual(conditions[1].range.gte, 1)
        self.assertEqual(conditions[1].range.lte, 3)

    def test_rejects_neighbor_from_another_document(self):
        seed = ContextChunk("DOC:1", "DOC", 1, None, [], [], [], "texto", role="seed")
        decision = validate_neighbor(
            seed,
            {"document_id": "OTHER", "chunk_index": 2, "text": "texto"},
            "consulta",
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "different_or_missing_document")

    def test_includes_unknown_structure_when_consecutive(self):
        client = MagicMock()
        client.scroll.return_value = ([point("DOC:2", "DOC", 2, "Continuación sin marca.")], None)
        result = build_context(
            "consulta",
            [seed_chunk(text="Texto sin estructura.")],
            client,
            config(),
        )
        neighbor = next(chunk for chunk in result.chunks if chunk.chunk_id == "DOC:2")
        self.assertEqual(neighbor.structure_status, "unknown")
        self.assertIn("consecutive_structure_unknown", neighbor.inclusion_reason)

    def test_rejects_explicit_transition_to_another_article(self):
        client = MagicMock()
        client.scroll.return_value = (
            [point("DOC:2", "DOC", 2, "Art. 19.- Otro contenido.")],
            None,
        )
        result = build_context(
            "consulta",
            [seed_chunk(text="Art. 18.- Contenido.")],
            client,
            config(),
        )
        self.assertNotIn("DOC:2", [chunk.chunk_id for chunk in result.chunks])
        self.assertEqual(
            result.metadata["discarded_chunks"][0]["reason"],
            "explicit_structural_boundary",
        )

    def test_includes_list_items_adjacent_to_article(self):
        client = MagicMock()
        client.scroll.return_value = (
            [point("DOC:2", "DOC", 2, "- a) Primer literal.", item_types=["list_item"])],
            None,
        )
        result = build_context(
            "consulta",
            [seed_chunk(text="Art. 18.- Requisitos:")],
            client,
            config(),
        )
        neighbor = next(chunk for chunk in result.chunks if chunk.chunk_id == "DOC:2")
        self.assertEqual(neighbor.structural_type, "list")
        self.assertEqual(neighbor.role, "expanded")

    def test_expands_a_list_fragmented_across_multiple_chunks(self):
        records = [
            point("DOC:1", "DOC", 1, "Art. 18.- Requisitos:"),
            point("DOC:2", "DOC", 2, "- a) Primer literal.", item_types=["list_item"]),
            point("DOC:3", "DOC", 3, "- b) Segundo literal.", item_types=["list_item"]),
            point("DOC:4", "DOC", 4, "Art. 19.- Otro artículo."),
        ]
        client = MagicMock()

        def scroll(**kwargs):
            conditions = kwargs["scroll_filter"].must
            lower = conditions[1].range.gte
            upper = conditions[1].range.lte
            matches = [
                record for record in records
                if lower <= record.payload["chunk_index"] <= upper
                and record.payload["document_id"] == conditions[0].match.value
            ]
            return matches[:kwargs["limit"]], None

        client.scroll.side_effect = scroll
        result = build_context(
            "consulta",
            [seed_chunk(text="Art. 18.- Requisitos:")],
            client,
            config(),
        )
        self.assertEqual(
            [chunk.chunk_id for chunk in result.chunks],
            ["DOC:00001", "DOC:2", "DOC:3"],
        )

    def test_complementary_reform_note_is_conditional(self):
        reform = point(
            "DOC:2",
            "DOC",
            2,
            "Nota: Disposición agregada por reforma.",
            headings=["DISPOSICIÓN REFORMATORIA"],
        )
        client = MagicMock()
        client.scroll.return_value = ([reform], None)
        without_justification = build_context(
            "¿Cuál es el artículo?",
            [seed_chunk(text="Art. 18.- Contenido.")],
            client,
            config(),
        )
        self.assertNotIn("DOC:2", [chunk.chunk_id for chunk in without_justification.chunks])
        self.assertEqual(
            without_justification.metadata["discarded_chunks"][0]["reason"],
            "conditional_complement_not_justified_by_query",
        )

        with_justification = build_context(
            "¿Cuál fue la reforma?",
            [seed_chunk(text="Art. 18.- Contenido.")],
            client,
            config(),
        )
        complementary = next(chunk for chunk in with_justification.chunks if chunk.chunk_id == "DOC:2")
        self.assertEqual(complementary.role, "complementary")

    def test_complementary_concordance_is_conditional(self):
        concordance = point(
            "DOC:2",
            "DOC",
            2,
            "Concordancias:\nConstitución, Art. 10.",
            headings=["Concordancias:"],
        )
        client = MagicMock()
        client.scroll.return_value = ([concordance], None)
        result = build_context(
            "¿Qué concordancias tiene?",
            [seed_chunk(text="Art. 18.- Contenido.")],
            client,
            config(),
        )
        self.assertEqual(
            next(chunk for chunk in result.chunks if chunk.chunk_id == "DOC:2").role,
            "complementary",
        )

    def test_deduplicates_and_preserves_seed_traceability(self):
        first = ContextChunk(
            "DOC:2", "DOC", 2, None, [], [], [], "vecino",
            included_by_seed_ids=["DOC:1"], inclusion_reason=["reason-a"],
        )
        duplicate = ContextChunk(
            "DOC:2", "DOC", 2, None, [], [], [], "vecino",
            included_by_seed_ids=["DOC:3"], inclusion_reason=["reason-b"],
        )
        merged, duplicates, duplicate_records = deduplicate_and_sort([first, duplicate])
        self.assertEqual(duplicates, 1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].included_by_seed_ids, ["DOC:1", "DOC:3"])
        self.assertEqual(merged[0].inclusion_reason, ["reason-a", "reason-b"])
        self.assertEqual(duplicate_records[0]["reason"], "duplicate_chunk_id")

    def test_merges_overlapping_seed_windows_and_keeps_all_seed_ids(self):
        shared = point("DOC:2", "DOC", 2, "vecino sin estructura")
        client = MagicMock()
        client.scroll.return_value = ([shared], None)
        ranked = [
            seed_chunk("DOC:1", "DOC", 1, "primer seed"),
            seed_chunk("DOC:3", "DOC", 3, "segundo seed", 0.8),
        ]
        result = build_context("consulta", ranked, client, config())
        neighbors = [chunk for chunk in result.chunks if chunk.chunk_id == "DOC:2"]
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0].included_by_seed_ids, ["DOC:1", "DOC:3"])

    def test_sorts_by_document_then_chunk_index(self):
        chunks = [
            ContextChunk("B:1", "B", 1, None, [], [], [], "b"),
            ContextChunk("A:3", "A", 3, None, [], [], [], "a3"),
            ContextChunk("A:1", "A", 1, None, [], [], [], "a1"),
        ]
        ordered, _, _ = deduplicate_and_sort(chunks)
        self.assertEqual([chunk.chunk_id for chunk in ordered], ["A:1", "A:3", "B:1"])

    def test_token_budget_preserves_seeds_and_records_excluded_chunks(self):
        chunks = [
            ContextChunk("DOC:1", "DOC", 1, None, [], [], [], "semilla", role="seed"),
            ContextChunk("DOC:2", "DOC", 2, None, [], [], [], "ampliación", role="expanded"),
        ]
        selected, excluded, metadata = apply_token_budget(
            chunks,
            {"max_tokens": 1, "token_counter": lambda text: len(text.split())},
        )
        self.assertEqual([chunk.chunk_id for chunk in selected], ["DOC:1"])
        self.assertEqual([chunk.chunk_id for chunk in excluded], ["DOC:2"])
        self.assertEqual(metadata["token_count"], 1)
        self.assertFalse(metadata["token_count_is_estimated"])

    def test_handles_missing_payload_fields_and_estimates_tokens(self):
        result = build_context(
            "consulta",
            [(SimpleNamespace(id="missing", payload=None), 0.2)],
            empty_client(),
            config(),
        )
        self.assertEqual(result.metadata["metrics"]["seed_count"], 1)
        self.assertTrue(result.metadata["metrics"]["token_count_is_estimated"])
        self.assertEqual(result.metadata["warnings"].__len__(), 1)

    def test_reports_distinct_context_stage_timings(self):
        result = build_context("consulta", [seed_chunk()], empty_client(), config())
        timings = result.metadata["timings_seconds"]
        for stage in (
            "seed_selection",
            "neighbor_expansion",
            "structural_validation",
            "deduplication_and_merging",
            "token_budget",
            "context_assembly",
            "total",
        ):
            self.assertGreaterEqual(timings[stage], 0)

    def test_exports_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            export_validation_json(path, {"query": "consulta", "discarded_chunks": []})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "query": "consulta",\n  "discarded_chunks": []\n}\n')

    def test_surfaces_model_initialization_errors(self):
        with patch("validate_context_pipeline.load_config", return_value={"e5_model_path": Path("missing")}):
            with patch("validate_context_pipeline.load_e5", side_effect=FileNotFoundError("missing E5")):
                with self.assertRaisesRegex(FileNotFoundError, "missing E5"):
                    main(["consulta"])

    def test_surfaces_qdrant_initialization_errors(self):
        config_data = {
            "e5_model_path": Path("e5"),
            "reranker_model_path": Path("bge"),
            "collection_name": COLLECTION,
        }
        with patch("validate_context_pipeline.load_config", return_value=config_data):
            with patch("validate_context_pipeline.load_e5", return_value=object()):
                with patch("validate_context_pipeline.load_reranker", return_value=object()):
                    with patch(
                        "validate_context_pipeline.open_client",
                        side_effect=RuntimeError("Qdrant unavailable"),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "Qdrant unavailable"):
                            main(["consulta"])

    def test_reranker_is_enabled_by_default_and_can_be_disabled(self):
        parser = _parser()
        self.assertTrue(parser.parse_args(["consulta"]).rerank)
        self.assertFalse(parser.parse_args(["consulta", "--no-rerank"]).rerank)
        self.assertTrue(parser.parse_args(["consulta", "--rerank"]).rerank)

    def test_no_rerank_skips_model_and_uses_rrf_candidates(self):
        config_data = {
            "e5_model_path": Path("e5"),
            "reranker_model_path": Path("bge"),
            "collection_name": COLLECTION,
        }
        client = empty_client()
        candidate = point("DOC:1", "DOC", 1, "Texto candidato.")
        with patch("validate_context_pipeline.load_config", return_value=config_data):
            with patch("validate_context_pipeline.load_e5", return_value=object()):
                with patch("validate_context_pipeline.load_reranker") as load_model:
                    with patch("validate_context_pipeline.open_client", return_value=client):
                        with patch("validate_context_pipeline.encode_query", return_value=[0.1]):
                            with patch(
                                "validate_context_pipeline.hybrid_search",
                                return_value=[candidate],
                            ):
                                with patch("validate_context_pipeline.rerank") as rerank_model:
                                    with patch(
                                        "validate_context_pipeline.build_context",
                                        wraps=build_context,
                                    ) as build:
                                        with redirect_stdout(StringIO()):
                                            self.assertEqual(
                                                main(["consulta", "--no-rerank"]),
                                                0,
                                            )
        load_model.assert_not_called()
        rerank_model.assert_not_called()
        self.assertEqual(
            build.call_args.args[1],
            [(candidate, None)],
        )
        self.assertEqual(
            build.call_args.args[3]["context_builder"]["seed_count"],
            3,
        )
        client.close.assert_called_once()

    def test_structure_detector_handles_decimal_articles_and_unknown_metadata(self):
        self.assertEqual(
            detect_structure({"text": "Artículo 18.2.- Texto", "headings": [], "item_types": []})["id"],
            "18.2",
        )
        self.assertEqual(detect_structure({"text": "sin metadata"})["status"], "unknown")

    def test_pdf_docx_optional_metadata_remains_empty(self):
        pdf = point("PDF:1", "PDF", 1, "Texto", page_numbers=[3], headings=["Capítulo I"])
        docx = point("DOCX:1", "DOCX", 1, "Texto", page_numbers=[], headings=[])
        result = build_context("consulta", [(pdf, 0.9), (docx, 0.8)], empty_client(), config())
        by_id = {chunk.chunk_id: chunk for chunk in result.chunks}
        self.assertEqual(by_id["PDF:1"].page_numbers, [3])
        self.assertEqual(by_id["DOCX:1"].page_numbers, [])
        self.assertEqual(by_id["DOCX:1"].headings, [])


if __name__ == "__main__":
    unittest.main()

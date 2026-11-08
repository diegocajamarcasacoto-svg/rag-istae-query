"""Pruebas de la recuperación jerárquica (query.hierarchy) y del modo hierarchical de query_rag."""

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

from query.hierarchy import (
    HIERARCHY_SECTION_TITLE,
    RELATION_COMPATIBLE,
    RELATION_DEVELOPS,
    catalog_from_dict,
    expand_with_hierarchy,
    hierarchy_targets,
    load_catalog,
    resolve_relations,
    select_hierarchy_results,
)
from query.units import Unit, UnitIndex, build_unit_context

import query_rag
from query import pipeline

CATALOG = {
    "source": "matriz.xlsx",
    "source_sha256": "abc",
    "rank_labels": {"2": "Ley orgánica", "3": "Reglamento nacional", "5": "Estatuto", "6": "Reglamento interno"},
    "documents": [
        {"document_id": "EXT01", "code": "LOES", "name": "LOES", "rank": 2, "rank_label": "Ley orgánica"},
        {"document_id": "EXT02", "code": "RRA", "name": "RRA", "rank": 3, "rank_label": "Reglamento nacional"},
        {"document_id": "INT01", "code": "EST", "name": "Estatuto", "rank": 5, "rank_label": "Estatuto"},
        {"document_id": "INT08", "code": "RTG", "name": "Titulación", "rank": 6, "rank_label": "Reglamento interno"},
        {"document_id": "INT07", "code": "REEI", "name": "Evaluación", "rank": 6, "rank_label": "Reglamento interno"},
    ],
    "relations": [
        {"source_document_id": "INT08", "relation": RELATION_DEVELOPS, "target_document_id": "INT01"},
        {"source_document_id": "INT08", "relation": RELATION_COMPATIBLE, "target_document_id": "EXT02"},
        {"source_document_id": "INT08", "relation": RELATION_COMPATIBLE, "target_document_id": "INT07"},  # mismo rango
        {"source_document_id": "INT01", "relation": RELATION_DEVELOPS, "target_document_id": "EXT01"},
        {"source_document_id": "INT01", "relation": RELATION_COMPATIBLE, "target_document_id": "INT08"},  # hacia abajo
        {"source_document_id": "EXT02", "relation": RELATION_DEVELOPS, "target_document_id": "EXT01"},
        {"source_document_id": "EXT02", "relation": RELATION_DEVELOPS, "target_document_id": None},  # fuera del corpus
    ],
}


def _catalog():
    return catalog_from_dict(CATALOG)


def _point(chunk_id: str, score: float = 0.5) -> SimpleNamespace:
    document_id = chunk_id.split(":")[0]
    return SimpleNamespace(
        payload={"chunk_id": chunk_id, "document_id": document_id, "source_file": f"{document_id}.pdf",
                 "page_numbers": [], "text": f"texto {chunk_id}"},
        score=score,
    )


def _index() -> UnitIndex:
    return UnitIndex([
        Unit("INT08-art12", "INT08", "articulo", "Art. 12", "Egresado", "", ("INT08:00039",), "Artículo 12. Egresado."),
        Unit("INT01-art5", "INT01", "articulo", "Art. 5", "Fines", "", ("INT01:00010",), "Artículo 5. Fines."),
        Unit("INT01-art6", "INT01", "articulo", "Art. 6", "", "", ("INT01:00011",), "Artículo 6. Otro."),
        Unit("EXT02-art30", "EXT02", "articulo", "Art. 30", "Titulación", "", ("EXT02:00100", "EXT02:00101"),
             "Artículo 30. Titulación."),
        Unit("EXT01-art1", "EXT01", "articulo", "Art. 1", "", "", ("EXT01:00001",), "Artículo 1. Ámbito."),
    ])


class CatalogTests(unittest.TestCase):
    def test_relations_without_document_id_are_ignored(self) -> None:
        catalog = _catalog()
        self.assertEqual(len(catalog.documents), 5)
        self.assertEqual(len(catalog.relations), 6)
        self.assertEqual(catalog.document("INT08").rank, 6)

    def test_resolve_relation_aliases(self) -> None:
        self.assertEqual(resolve_relations(["desarrolla", " Compatible "]), (RELATION_DEVELOPS, RELATION_COMPATIBLE))
        with self.assertRaises(ValueError):
            resolve_relations(["deroga"])
        with self.assertRaises(ValueError):
            resolve_relations([""])

    def test_load_catalog_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalogo.json"
            path.write_text(json.dumps(CATALOG), encoding="utf-8")
            self.assertEqual(load_catalog(path).source_sha256, "abc")
            with self.assertRaises(FileNotFoundError):
                load_catalog(Path(directory) / "no.json")


class HierarchyTargetsTests(unittest.TestCase):
    def test_only_strictly_superior_documents_ordered_by_rank(self) -> None:
        targets, warnings = hierarchy_targets(_catalog(), ["INT08"], (RELATION_DEVELOPS, RELATION_COMPATIBLE), 2)
        self.assertEqual([t.document_id for t in targets], ["EXT01", "EXT02", "INT01"])
        self.assertNotIn("INT07", [t.document_id for t in targets])  # mismo rango
        self.assertEqual(warnings, [])
        ext01 = targets[0]
        self.assertEqual(ext01.depth, 2)
        self.assertEqual(set(ext01.via), {("INT01", RELATION_DEVELOPS), ("EXT02", RELATION_DEVELOPS)})

    def test_depth_limit_and_relation_filter(self) -> None:
        targets, _ = hierarchy_targets(_catalog(), ["INT08"], (RELATION_DEVELOPS,), 1)
        self.assertEqual([t.document_id for t in targets], ["INT01"])
        targets, _ = hierarchy_targets(_catalog(), ["INT08"], (RELATION_DEVELOPS,), 2)
        self.assertEqual([t.document_id for t in targets], ["EXT01", "INT01"])

    def test_seeds_are_not_targets_and_cycles_terminate(self) -> None:
        targets, _ = hierarchy_targets(_catalog(), ["INT08", "INT01"], (RELATION_DEVELOPS, RELATION_COMPATIBLE), 5)
        self.assertEqual([t.document_id for t in targets], ["EXT01", "EXT02"])

    def test_unknown_seed_produces_warning(self) -> None:
        targets, warnings = hierarchy_targets(_catalog(), ["INT99"], (RELATION_DEVELOPS,), 2)
        self.assertEqual(targets, [])
        self.assertEqual(len(warnings), 1)

    def test_invalid_depth(self) -> None:
        with self.assertRaises(ValueError):
            hierarchy_targets(_catalog(), ["INT08"], (RELATION_DEVELOPS,), 0)


class SelectHierarchyResultsTests(unittest.TestCase):
    def test_caps_exclusions_and_min_score(self) -> None:
        ranked = [
            (_point("INT08:00039"), 0.99),   # ya en el contexto base
            (_point("EXT02:00100"), 0.9),
            (_point("EXT02:00101"), 0.8),    # misma unidad elegida: se conserva
            (_point("INT01:00010"), 0.7),
            (_point("INT01:00011"), 0.6),    # tope por norma
            (_point("INT01:00011"), 0.55),   # descarte repetido: una sola entrada
            (_point("EXT01:00001"), 0.05),   # bajo el mínimo
        ]
        selected, skipped = select_hierarchy_results(
            ranked, _index(), exclude_unit_ids=["INT08-art12"], max_units=3, max_per_document=1, min_score=0.1,
        )
        self.assertEqual([p.payload["chunk_id"] for p, _ in selected], ["EXT02:00100", "EXT02:00101", "INT01:00010"])
        self.assertEqual(
            [(item["unit_id"], item["reason"]) for item in skipped],
            [("INT08-art12", "already_in_base_context"), ("INT01-art6", "max_per_document"),
             ("EXT01-art1", "below_min_score")],
        )

    def test_max_units(self) -> None:
        ranked = [(_point("EXT02:00100"), 0.9), (_point("INT01:00010"), 0.8), (_point("EXT01:00001"), 0.7)]
        selected, skipped = select_hierarchy_results(ranked, _index(), max_units=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(skipped[0]["reason"], "max_units")


class ExpandWithHierarchyTests(unittest.TestCase):
    def _base(self):
        return build_unit_context("pregunta", [(_point("INT08:00039"), 0.9)], _index())

    def test_adds_superior_units_after_base_with_rank(self) -> None:
        searched = []

        def search(document_ids):
            searched.append(document_ids)
            return [(_point("EXT02:00100"), 0.8), (_point("INT01:00010"), 0.6)]

        result = expand_with_hierarchy(
            "pregunta", self._base(), _index(), _catalog(), search,
            relations=(RELATION_DEVELOPS, RELATION_COMPATIBLE), max_depth=2,
        )
        self.assertEqual(searched, [["EXT01", "EXT02", "INT01"]])
        self.assertEqual([c.chunk_id for c in result.chunks], ["INT08-art12", "EXT02-art30", "INT01-art5"])
        self.assertEqual([c.role for c in result.chunks], ["seed", "hierarchy", "hierarchy"])
        self.assertIn("hierarchy", result.chunks[1].inclusion_reason)
        self.assertIn("· Jerarquía 6: Reglamento interno", result.chunks[0].text.splitlines()[0])
        self.assertIn("· Jerarquía 3: Reglamento nacional", result.chunks[1].text.splitlines()[0])
        base_part, hierarchy_part = result.context.split(HIERARCHY_SECTION_TITLE)
        self.assertIn("Artículo 12", base_part)
        self.assertIn("Artículo 30", hierarchy_part)
        metadata = result.metadata
        self.assertEqual(metadata["retrieval_mode"], "hierarchical")
        self.assertEqual(metadata["metrics"]["hierarchy_units"], 2)
        self.assertEqual(metadata["metrics"]["final_chunk_count"], 3)
        self.assertEqual([u["unit_id"] for u in metadata["hierarchy"]["units"]], ["EXT02-art30", "INT01-art5"])
        self.assertEqual(metadata["hierarchy"]["units"][0]["rank"], 3)
        self.assertEqual(metadata["hierarchy"]["seed_documents"], ["INT08"])

    def test_without_targets_does_not_search(self) -> None:
        base = build_unit_context("pregunta", [(_point("EXT01:00001"), 0.9)], _index())
        result = expand_with_hierarchy("pregunta", base, _index(), _catalog(),
                                       search=mock.Mock(side_effect=AssertionError("no debe buscar")))
        self.assertEqual([c.chunk_id for c in result.chunks], ["EXT01-art1"])
        self.assertNotIn(HIERARCHY_SECTION_TITLE, result.context)
        self.assertEqual(result.metadata["hierarchy"]["targets"], [])
        self.assertTrue(any("Sin normas superiores" in w for w in result.metadata["warnings"]))

    def test_hierarchy_budget_is_separate_from_base(self) -> None:
        result = expand_with_hierarchy(
            "pregunta", self._base(), _index(), _catalog(),
            lambda ids: [(_point("EXT02:00100"), 0.8), (_point("INT01:00010"), 0.6)],
            relations=(RELATION_DEVELOPS, RELATION_COMPATIBLE), max_tokens=12,
        )
        # La primera unidad de jerarquía siempre entra; la segunda no cabe.
        self.assertEqual([c.chunk_id for c in result.chunks], ["INT08-art12", "EXT02-art30"])
        self.assertEqual(result.metadata["hierarchy"]["excluded_units"][0]["unit_id"], "INT01-art5")


class QueryRagHierarchicalTests(unittest.TestCase):
    def test_hierarchical_without_generation_end_to_end_with_mocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            units_path = Path(directory) / "units.jsonl"
            units_path.write_text("".join(json.dumps({
                "unit_id": unit.unit_id, "document_id": unit.document_id, "kind": unit.kind,
                "label": unit.label, "title": unit.title, "chunk_ids": list(unit.chunk_ids), "text": unit.text,
            }) + "\n" for unit in _index()._units.values()), encoding="utf-8")
            catalog_path = Path(directory) / "catalogo.json"
            catalog_path.write_text(json.dumps(CATALOG), encoding="utf-8")
            log_dir = Path(directory) / "logs"
            calls = []

            def _search(client, config, vector, text, prefetch_k, candidate_k, rrf_k, document_ids=None):
                calls.append(document_ids)
                if document_ids is None:
                    return [_point("INT08:00039", 0.9)]
                return [_point(f"{doc}:{'00100' if doc == 'EXT02' else '00010' if doc == 'INT01' else '00001'}",
                               0.5) for doc in document_ids]

            patches = [
                mock.patch.object(query_rag, "load_config",
                                  return_value={"collection_name": "c", "reranker_model_path": "bge",
                                                "units_path": str(units_path), "catalog_path": str(catalog_path)}),
                mock.patch.object(query_rag, "load_generation_config", side_effect=AssertionError("sin generación")),
                mock.patch.object(query_rag, "generate_answer", side_effect=AssertionError("sin generación")),
                mock.patch.object(query_rag, "load_e5", return_value=object()),
                mock.patch.object(query_rag, "open_client", return_value=mock.MagicMock()),
                # La rama units corre en query.pipeline (compartido con el servicio /retrieve).
                mock.patch.object(pipeline, "encode_query", return_value=[0.0]),
                mock.patch.object(pipeline, "hybrid_search", side_effect=_search),
                mock.patch.dict(os.environ, {}, clear=True),
            ]
            for patch in patches:
                patch.start()
                self.addCleanup(patch.stop)
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                code = query_rag.main(["pregunta", "--no-rerank", "--retrieval-mode", "hierarchical",
                                       "--hierarchy-search", "per-document", "--no-generation",
                                       "--show-context", "--log-dir", str(log_dir)])
            with (log_dir / "runs_index.csv").open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle, delimiter=";"))
            detail = json.loads(next((log_dir / "runs").glob("*.json")).read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(calls, [None, ["EXT01"], ["EXT02"], ["INT01"]])
        self.assertEqual(row["retrieval_mode"], "hierarchical")
        self.assertEqual(row["expansion"], "units")
        self.assertEqual(row["context_chunk_ids"].split("|")[0], "INT08-art12")
        self.assertEqual(len(row["hierarchy_unit_ids"].split("|")), 3)
        self.assertEqual(row["provider"], "")
        self.assertIsNone(detail["generation_result"])
        self.assertEqual(detail["context_result"]["metadata"]["retrieval_mode"], "hierarchical")
        self.assertIsNone(detail["generation_config"])
        self.assertEqual(len(detail["retrieval_config"]["catalog_sha256"]), 64)
        # La jerarquía se mide aparte del Context Builder.
        timings = detail["timings_seconds"]
        self.assertIsNotNone(timings["context_builder"])
        self.assertIsNotNone(timings["hierarchy"])
        self.assertLessEqual(timings["hierarchy_search"], timings["hierarchy"])
        self.assertNotEqual(row["hierarchy_ms"], "")
        printed = out.getvalue()
        self.assertIn("=== ESCALA DE JERARQUÍA NORMATIVA", printed)  # presentación por documentos
        self.assertIn("· incluida por jerarquía ===", printed)
        self.assertIn("=== RELACIONES ENTRE NORMAS ===", printed)
        self.assertIn("Generación: omitida", printed)

    def test_hierarchical_rejects_neighbors_expansion(self) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                query_rag.main(["pregunta", "--retrieval-mode", "hierarchical", "--expansion", "neighbors",
                                "--no-generation", "--no-log"])

    def test_unknown_relation_is_rejected(self) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                query_rag.main(["pregunta", "--retrieval-mode", "hierarchical",
                                "--hierarchy-relations", "deroga", "--no-generation", "--no-log"])


if __name__ == "__main__":
    unittest.main()

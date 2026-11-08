"""Pruebas del pipeline compartido B/C (query.pipeline) y del motor residente (query.engine)."""

import unittest
from dataclasses import replace
from unittest import mock

from query import pipeline
from query.baseline import LB1
from query.engine import RetrievalEngine
from query.pipeline import run_units_pipeline

from test_hierarchy import _catalog, _index, _point

QUESTION = "¿Qué requisitos hay para egresar?"


def _search(calls):
    def search(client, config, vector, text, prefetch_k, candidate_k, rrf_k, document_ids=None):
        calls.append({"document_ids": document_ids, "prefetch_k": prefetch_k, "candidate_k": candidate_k,
                      "rrf_k": rrf_k})
        if document_ids is None or document_ids == ["INT08"]:
            return [_point("INT08:00039", 0.9), _point("INT01:00011", 0.4)]
        return [_point(f"{doc}:{'00100' if doc == 'EXT02' else '00010' if doc == 'INT01' else '00001'}", 0.5)
                for doc in document_ids]
    return search


def _rerank(model, query, candidates, batch_size):
    # Invierte el orden RRF para comprobar que el rerank se aplica.
    return [(point, 1.0 - position / 10) for position, point in enumerate(reversed(list(candidates)))]


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = []
        for patch in (
            mock.patch.object(pipeline, "encode_query", return_value=[0.0]),
            mock.patch.object(pipeline, "hybrid_search", side_effect=_search(self.calls)),
            mock.patch.object(pipeline, "rerank_candidates", side_effect=_rerank),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _run(self, **kwargs):
        options = dict(e5=object(), reranker=object(), client=mock.MagicMock(), config={"collection_name": "c"},
                       unit_index=_index(), catalog=_catalog(), params=LB1)
        options.update(kwargs)
        return run_units_pipeline(QUESTION, **options)

    def test_hybrid_with_rerank_uses_baseline_and_reranked_order(self) -> None:
        result = self._run()
        self.assertEqual(self.calls[0]["document_ids"], None)
        self.assertEqual((self.calls[0]["prefetch_k"], self.calls[0]["candidate_k"], self.calls[0]["rrf_k"]),
                         (LB1.prefetch_k, LB1.candidate_k, LB1.rrf_k))
        self.assertEqual([c.chunk_id for c in result.context_result.chunks], ["INT01-art6", "INT08-art12"])
        self.assertIsNotNone(result.timings["reranking"])
        self.assertNotIn("hierarchy", result.timings)
        self.assertEqual(result.context_result.metadata["presentation"], "documents")

    def test_without_rerank_keeps_rrf_order_and_needs_no_model(self) -> None:
        result = self._run(reranker=None, rerank=False)
        self.assertEqual([c.chunk_id for c in result.context_result.chunks], ["INT08-art12", "INT01-art6"])
        self.assertIsNone(result.timings["reranking"])
        # Sin rerank, el score de cada artículo es el de la fusión RRF de su mejor chunk.
        self.assertEqual(result.context_result.chunks[0].reranker_score, 0.9)

    def test_rerank_requires_model(self) -> None:
        with self.assertRaises(ValueError):
            self._run(reranker=None, rerank=True)

    def test_document_ids_filter_base_search(self) -> None:
        self._run(document_ids=["INT08"])
        self.assertEqual(self.calls[0]["document_ids"], ["INT08"])

    def test_hierarchy_adds_superior_units(self) -> None:
        result = self._run(rerank=False, reranker=None, hierarchy=True,
                           params=replace(LB1, top_k=1))
        roles = {c.chunk_id: c.role for c in result.context_result.chunks}
        self.assertEqual(roles["INT08-art12"], "seed")
        self.assertIn("hierarchy", roles.values())
        self.assertIn("hierarchy", result.timings)
        self.assertIn("=== ESCALA DE JERARQUÍA NORMATIVA", result.context_result.context)
        # Búsqueda jerárquica conjunta, filtrada por las normas superiores.
        self.assertEqual(sorted(self.calls[1]["document_ids"]), ["EXT01", "EXT02", "INT01"])
        self.assertEqual(self.calls[1]["candidate_k"], LB1.hierarchy_candidate_k)

    def test_hierarchy_requires_catalog(self) -> None:
        with self.assertRaises(ValueError):
            self._run(hierarchy=True, catalog=None)


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        for patch in (
            mock.patch.object(pipeline, "encode_query", return_value=[0.0]),
            mock.patch.object(pipeline, "hybrid_search", side_effect=_search([])),
            mock.patch.object(pipeline, "rerank_candidates", side_effect=_rerank),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.engine = RetrievalEngine(
            config={"collection_name": "c"}, e5=object(), reranker=object(), client=mock.MagicMock(),
            unit_index=_index(), catalog=_catalog(), index_version={"collection": "c"},
        )

    def test_describes_units_for_the_api(self) -> None:
        output = self.engine.retrieve(QUESTION, rerank=False, hierarchy=True)
        first = output.units[0]
        self.assertEqual(first["unit_id"], "INT08-art12")
        self.assertEqual((first["document_name"], first["rank"], first["origin"]), ("Titulación", 6, "base"))
        self.assertEqual(first["text"], "Artículo 12. Egresado.")  # sin la línea de encabezado
        self.assertEqual(first["matched_chunk_ids"], ["INT08:00039"])
        self.assertIn("hierarchy", {unit["origin"] for unit in output.units})
        self.assertEqual(output.tokens_estimated, output.context_result.metadata["metrics"]["token_count"])
        self.assertIn("total", output.timings)
        self.assertEqual(output.debug["candidates"][0]["chunk_id"], "INT08:00039")

    def test_known_documents_and_info(self) -> None:
        self.assertIn("EXT01", self.engine.document_ids)
        self.assertEqual(self.engine.info()["baseline"], "LB-1")


if __name__ == "__main__":
    unittest.main()

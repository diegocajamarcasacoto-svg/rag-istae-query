"""Pruebas del servicio /retrieve (FastAPI TestClient con un motor falso, sin modelos ni Qdrant)."""

import csv
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from query.baseline import LB1
from query.engine import RetrievalOutput
from query.service.app import create_app
from query.service.settings import ServiceSettings, load_settings

KEY = "clave-de-prueba-secreta"
HEADERS = {"X-API-Key": KEY}


class FakeEngine:
    params = LB1
    index_version = {"collection": "c", "units_sha256": "u", "catalog_sha256": "k"}
    load_timings = {"e5_load": 1.0}
    document_ids = frozenset({"INT08", "EXT02"})

    def __init__(self, fail: bool = False) -> None:
        self.calls = []
        self.fail = fail

    def info(self):
        return {"baseline": "LB-1", "units": 2, "documents": 2, "index_version": self.index_version}

    def retrieve(self, question, *, rerank=True, hierarchy=False, document_ids=None):
        self.calls.append((question, rerank, hierarchy, document_ids))
        if self.fail:
            raise RuntimeError("Qdrant no responde")
        units = [{"unit_id": "INT08-art12", "document_id": "INT08", "document_name": "Titulación", "rank": 6,
                  "label": "Art. 12", "title": "", "text": "Artículo 12.", "partial": False, "score": 0.9,
                  "origin": "base", "matched_chunk_ids": ["INT08:00039"]}]
        if hierarchy:
            units.append({**units[0], "unit_id": "EXT02-art43", "document_id": "EXT02", "rank": 3,
                          "origin": "hierarchy", "matched_chunk_ids": ["EXT02:00100"]})
        return RetrievalOutput(
            units=units, context="=== [INT08] Titulación ===\n[INT08-art12] Artículo 12.",
            tokens_estimated=12, warnings=[], timings={"query_embedding": 0.1, "reranking": 0.2},
            debug={"candidates": [{"chunk_id": "INT08:00039", "rrf_score": 0.03}]},
        )


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_dir = Path(self.tmp.name)
        self.engine = FakeEngine()
        self.client = self._client(self.engine)

    def _client(self, engine, **settings):
        values = dict(retrieval_config={}, api_key=KEY, auth_disabled=False, log_dir=self.log_dir, cache_size=8)
        values.update(settings)

        def factory(_settings):
            if isinstance(engine, Exception):
                raise engine
            return engine

        client = TestClient(create_app(ServiceSettings(**values), engine_factory=factory))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def _post(self, **body):
        return self.client.post("/retrieve", json={"question": "¿Qué requisitos hay para egresar?", **body},
                                headers=HEADERS)

    def test_health_without_secrets(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["baseline"], "LB-1")
        self.assertNotIn(KEY, response.text)

    def test_requires_api_key(self) -> None:
        self.assertEqual(self.client.post("/retrieve", json={"question": "q"}).status_code, 401)
        self.assertEqual(self.client.post("/retrieve", json={"question": "q"},
                                          headers={"X-API-Key": "otra"}).status_code, 401)

    def test_auth_can_be_disabled_for_local_tests(self) -> None:
        client = self._client(FakeEngine(), api_key=None, auth_disabled=True)
        self.assertEqual(client.post("/retrieve", json={"question": "q"}).status_code, 200)

    def test_contract_for_b_and_c(self) -> None:
        b = self._post().json()
        self.assertEqual(b["contract_version"], "1.0")
        self.assertEqual(b["baseline"], "LB-1")
        self.assertEqual(b["mode"], {"rerank": True, "hierarchy": False, "architecture": "B"})
        self.assertEqual(b["units"][0]["unit_id"], "INT08-art12")
        self.assertIn("[INT08-art12]", b["context"])
        self.assertFalse(b["cached"])
        self.assertNotIn("debug", b)
        c = self._post(hierarchy=True, rerank=False).json()
        self.assertEqual(c["mode"]["architecture"], "C")
        self.assertEqual(c["units"][-1]["origin"], "hierarchy")
        self.assertEqual(self.engine.calls[-1][1:3], (False, True))

    def test_follow_up_turns_reuse_cached_context(self) -> None:
        first = self._post(turn=1).json()
        second = self._post(turn=2, question_id="Q27").json()
        self.assertTrue(second["cached"])
        self.assertEqual(second["units"], first["units"])
        self.assertNotEqual(second["run_id"], first["run_id"])
        self.assertEqual(len(self.engine.calls), 1)
        self._post(hierarchy=True)  # otro modo: otra entrada del caché
        self.assertEqual(len(self.engine.calls), 2)

    def test_question_whitespace_is_normalized(self) -> None:
        self._post()
        self.client.post("/retrieve", json={"question": "  ¿Qué requisitos   hay para egresar?  "}, headers=HEADERS)
        self.assertEqual(len(self.engine.calls), 1)

    def test_validation_errors(self) -> None:
        cases = [
            {"question": "   "},
            {"question": "q", "document_ids": ["INT99"]},
            {"question": "q", "document_ids": []},
            {"question": "q", "top_k": 10},
            {"question": "q", "history": [{"role": "user", "content": "hola"}]},
            {"question": "q", "turn": 0},
        ]
        for body in cases:
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/retrieve", json=body, headers=HEADERS).status_code, 422)

    def test_document_ids_are_passed_to_engine(self) -> None:
        self._post(document_ids=["int08"])
        self.assertEqual(self.engine.calls[0][3], ["INT08"])

    def test_debug_details(self) -> None:
        debug = self._post(debug=True).json()["debug"]
        self.assertEqual(debug["candidates"][0]["chunk_id"], "INT08:00039")
        self.assertEqual(debug["timings_ms"]["execution"]["reranking"], 200)
        self.assertEqual(debug["baseline_params"]["candidate_k"], LB1.candidate_k)

    def test_runs_are_logged_without_secrets(self) -> None:
        self._post(question_id="Q27", tag="servicio", turn=1)
        self._post(question_id="Q27", tag="servicio", turn=2)
        with (self.log_dir / "runs_index.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=";"))
        self.assertEqual([row["question_id"] for row in rows], ["Q27", "Q27"])
        self.assertEqual(rows[0]["expansion"], "units")
        for path in self.log_dir.rglob("*"):
            if path.is_file():
                self.assertNotIn(KEY, path.read_text(encoding="utf-8-sig"))

    def test_engine_error_returns_500_with_run_id(self) -> None:
        client = self._client(FakeEngine(fail=True))
        with self.assertLogs("query.service", "ERROR"):  # la traza esperada no se imprime en la consola
            response = client.post("/retrieve", json={"question": "q"}, headers=HEADERS)
        self.assertEqual(response.status_code, 500)
        self.assertIn("run_id", response.json()["detail"])
        self.assertNotIn("Qdrant no responde", response.text)

    def test_engine_load_failure_returns_503(self) -> None:
        with self.assertLogs("query.service", "ERROR"):  # la traza esperada no se imprime en la consola
            client = self._client(FileNotFoundError("units.jsonl"))
        self.assertEqual(client.get("/health").status_code, 503)
        self.assertEqual(client.post("/retrieve", json={"question": "q"}, headers=HEADERS).status_code, 503)


class SettingsTests(unittest.TestCase):
    OVERRIDES = {"RAG_QDRANT_PATH": "q", "RAG_E5_MODEL_PATH": "e5", "RAG_RERANKER_MODEL_PATH": "bge",
                 "RAG_UNITS_PATH": "units.jsonl", "RAG_CATALOG_PATH": "catalogo.json", "RAG_COLLECTION": "c"}

    def setUp(self) -> None:
        # Configuración base vacía: las pruebas no dependen del config/local.json del equipo.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        empty = self.tmp / "vacio.json"
        empty.write_text("{}", encoding="utf-8")
        self.ENV = {"RAG_CONFIG": str(empty), **self.OVERRIDES}

    def test_config_with_utf8_bom_is_accepted(self) -> None:
        # PowerShell 5.1 (Set-Content -Encoding utf8) guarda el JSON con BOM.
        config = self.tmp / "local.json"
        config.write_text('{"collection_name": "desde-json"}', encoding="utf-8-sig")
        settings = load_settings({**self.OVERRIDES, "RAG_COLLECTION": "", "RAG_CONFIG": str(config),
                                  "RAG_API_KEY": KEY})
        self.assertEqual(settings.retrieval_config["collection_name"], "desde-json")

    def test_requires_api_key_unless_disabled(self) -> None:
        with self.assertRaises(ValueError):
            load_settings(dict(self.ENV))
        settings = load_settings({**self.ENV, "RAG_AUTH_DISABLED": "true"})
        self.assertTrue(settings.auth_disabled)

    def test_env_overrides_paths(self) -> None:
        settings = load_settings({**self.ENV, "RAG_API_KEY": KEY, "RAG_LOG_DIR": "-"})
        self.assertEqual(settings.retrieval_config["collection_name"], "c")
        self.assertTrue(settings.retrieval_config["units_path"].is_absolute())
        self.assertIsNone(settings.log_dir)
        self.assertNotIn(KEY, str(settings.describe()))

    def test_missing_config_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "vacio.json"
            empty.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(KeyError, "units_path"):
                load_settings({"RAG_CONFIG": str(empty), "RAG_API_KEY": KEY})


if __name__ == "__main__":
    unittest.main()

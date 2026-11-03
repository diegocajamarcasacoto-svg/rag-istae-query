"""Pruebas del adaptador de Gemini con un cliente simulado (sin red ni API key)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.long_context.llm import ChatMessage, LLMRequest  # noqa: E402
from query.long_context.providers.gemini import GeminiClient, GeminiConfigurationError  # noqa: E402
from query.long_context.settings import build_client, load_settings  # noqa: E402


class FakeModels:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def count_tokens(self, *, model, contents, config=None):
        self.calls.append(("count_tokens", {"model": model, "contents": contents}))
        return SimpleNamespace(total_tokens=12345)

    def get(self, *, model, config=None):
        return SimpleNamespace(input_token_limit=1_048_576)

    def _response(self):
        usage = SimpleNamespace(prompt_token_count=1000, cached_content_token_count=900, candidates_token_count=80, thoughts_token_count=40)
        return SimpleNamespace(text=" Respuesta [INT08-art12] ", usage_metadata=usage,
                               candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))], model_version="gemini-x-001")

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(("generate_content", {"model": model, "contents": contents, "config": config}))
        return self._response()

    def generate_content_stream(self, *, model, contents, config=None):
        self.calls.append(("generate_content_stream", {"model": model, "contents": contents, "config": config}))
        yield SimpleNamespace(text="Parte 1 ")
        yield SimpleNamespace(text=None)
        yield SimpleNamespace(text="parte 2")


class FakeCaches:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.deleted: list[str] = []

    def create(self, *, model, config=None):
        self.created.append({"model": model, "config": config})
        return SimpleNamespace(name=f"cachedContents/{len(self.created)}")

    def delete(self, *, name, config=None):
        self.deleted.append(name)


class FakeSDK:
    def __init__(self) -> None:
        self.models = FakeModels()
        self.caches = FakeCaches()


def make_request(**overrides) -> LLMRequest:
    values = dict(
        system="INSTRUCCIONES",
        context="=== [INT08] Reglamento ===\n[INT08-art12] Art. 12",
        question="  ¿Qué necesito?  ",
        history=(ChatMessage("user", "hola"), ChatMessage("assistant", "¿en qué ayudo?")),
        cache_key="v1",
    )
    values.update(overrides)
    return LLMRequest(**values)


class GeminiAdapterTests(unittest.TestCase):
    def test_prompt_layout_puts_documents_in_system_and_question_last(self) -> None:
        sdk = FakeSDK()
        client = GeminiClient("gemini-test", client=sdk)
        client.generate(make_request())
        _, call = sdk.models.calls[-1]
        config = call["config"]
        self.assertTrue(config.system_instruction.startswith("INSTRUCCIONES"))
        self.assertIn("<documentos>", config.system_instruction)
        self.assertEqual([c.role for c in call["contents"]], ["user", "model", "user"])
        self.assertIn("<pregunta>\n¿Qué necesito?\n</pregunta>", call["contents"][-1].parts[0].text)
        self.assertEqual(config.temperature, 0.0)
        self.assertIsNone(config.cached_content)

    def test_usage_and_finish_reason_are_mapped(self) -> None:
        response = GeminiClient("gemini-test", client=FakeSDK()).generate(make_request())
        self.assertEqual(response.text, "Respuesta [INT08-art12]")
        self.assertEqual((response.input_tokens, response.cached_input_tokens, response.output_tokens), (1000, 900, 80))
        self.assertEqual(response.finish_reason, "STOP")
        self.assertEqual(response.model, "gemini-x-001")
        self.assertEqual(response.extra["thoughts_tokens"], 40)

    def test_cache_is_created_once_per_corpus_version_and_deleted_on_close(self) -> None:
        sdk = FakeSDK()
        client = GeminiClient("gemini-test", client=sdk, use_cache=True, cache_ttl_seconds=900)
        client.generate(make_request())
        client.generate(make_request(question="otra"))
        self.assertEqual(len(sdk.caches.created), 1)
        created = sdk.caches.created[0]["config"]
        self.assertEqual(created.ttl, "900s")
        self.assertIn("<documentos>", created.system_instruction)
        config = sdk.models.calls[-1][1]["config"]
        self.assertEqual(config.cached_content, "cachedContents/1")
        self.assertIsNone(config.system_instruction)
        client.close()
        self.assertEqual(sdk.caches.deleted, ["cachedContents/1"])

    def test_count_tokens_includes_documents_and_window_is_read(self) -> None:
        sdk = FakeSDK()
        client = GeminiClient("gemini-test", client=sdk)
        self.assertEqual(client.count_tokens(make_request()), 12345)
        contents = sdk.models.calls[-1][1]["contents"]
        self.assertIn("<documentos>", contents[0].parts[0].text)
        self.assertEqual(client.context_window(), 1_048_576)

    def test_stream_skips_empty_chunks(self) -> None:
        pieces = list(GeminiClient("gemini-test", client=FakeSDK()).stream(make_request()))
        self.assertEqual(pieces, ["Parte 1 ", "parte 2"])

    def test_missing_api_key_gives_clear_error(self) -> None:
        env = "ISTAE_TEST_KEY_THAT_DOES_NOT_EXIST"
        os.environ.pop(env, None)
        with self.assertRaises(GeminiConfigurationError):
            GeminiClient("gemini-test", api_key_env=env).count_tokens(make_request())


class SettingsTests(unittest.TestCase):
    def test_relative_paths_resolve_from_settings_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_context.json"
            path.write_text(json.dumps({"provider": "gemini", "model": "gemini-test", "outputs_dir": "storage/outputs", "thinking_budget": 0}), encoding="utf-8")
            settings = load_settings(path)
            self.assertEqual(settings.outputs_dir, (Path(tmp) / "storage" / "outputs").resolve())
            self.assertEqual(settings.prompts_dir, (Path(tmp) / "prompts").resolve())
            self.assertEqual(settings.thinking_budget, 0)
            self.assertIsInstance(build_client(settings), GeminiClient)


if __name__ == "__main__":
    unittest.main()

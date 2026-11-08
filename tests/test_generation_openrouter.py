import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from query.context.models import ContextResult
from query.generation.errors import (
    GenerationConfigurationError,
    GenerationProviderError,
)
from query.generation.generation_config import GenerationConfig, load_generation_config
from query.generation.generator import generate_answer
from query.generation.models import GenerationPrompt
from query.generation.providers import create_provider
from query.generation.providers.gemini import GeminiProvider
from query.generation.providers.openai_compatible import (
    OpenAICompatibleProvider,
    parse_chat_completion,
)

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def _write_config(directory: Path, data: dict) -> Path:
    prompts = directory / "prompts"
    prompts.mkdir(exist_ok=True)
    for name in ("base_es.txt", "rag_es.txt", "jerarquia_es.txt", "recordatorio_es.txt"):
        (prompts / name).write_text(f"Reglas {name}", encoding="utf-8")
    path = directory / "generation.json"
    path.write_text(
        json.dumps({"system_prompt_path": "prompts", **data}), encoding="utf-8"
    )
    return path


def _config(**overrides) -> GenerationConfig:
    values = dict(
        provider="openrouter",
        model=MODEL,
        system_prompt_path=Path("system.txt"),
        temperature=0.2,
        max_output_tokens=256,
        timeout_seconds=30,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        extra_headers={"X-Title": "RAG ISTAE"},
        extra_body={"reasoning": {"exclude": True}},
    )
    values.update(overrides)
    return GenerationConfig(**values)


def _completion(**overrides) -> dict:
    data = {
        "id": "gen-1",
        "model": MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "  Respuesta.  "},
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 9},
    }
    data.update(overrides)
    return data


class _FakeResponse:
    def __init__(self, data, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data) if data is not None else "no-json"

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post(self, url, *, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self.response


PROMPT = GenerationPrompt(system_instruction="Reglas", user_content='{"question": "q"}')


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"})
class OpenRouterProviderTests(unittest.TestCase):
    def test_sends_chat_completion_and_normalizes_response(self) -> None:
        client = _FakeClient(_FakeResponse(_completion()))
        result = OpenAICompatibleProvider(client).generate(PROMPT, config=_config())

        call = client.calls[0]
        self.assertEqual(call["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(call["timeout"], 30)
        self.assertEqual(call["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(call["headers"]["X-Title"], "RAG ISTAE")
        self.assertEqual(call["json"]["model"], MODEL)
        self.assertEqual(call["json"]["max_tokens"], 256)
        self.assertEqual(call["json"]["temperature"], 0.2)
        self.assertEqual(call["json"]["reasoning"], {"exclude": True})
        self.assertEqual(
            call["json"]["messages"],
            [
                {"role": "system", "content": "Reglas"},
                {"role": "user", "content": '{"question": "q"}'},
            ],
        )
        self.assertEqual(result.text, "Respuesta.")
        self.assertEqual(result.model, MODEL)
        self.assertEqual((result.input_tokens, result.output_tokens), (40, 9))
        self.assertEqual(result.finish_reason, "stop")

    def test_extra_body_cannot_override_messages_or_model(self) -> None:
        client = _FakeClient(_FakeResponse(_completion()))
        config = _config(extra_body={"model": "otro", "messages": []})
        OpenAICompatibleProvider(client).generate(PROMPT, config=config)
        self.assertEqual(client.calls[0]["json"]["model"], MODEL)
        self.assertEqual(len(client.calls[0]["json"]["messages"]), 2)

    def test_http_429_raises_provider_error_with_hint(self) -> None:
        response = _FakeResponse({"error": {"message": "Rate limit", "code": 429}}, 429)
        with self.assertRaisesRegex(GenerationProviderError, "HTTP 429.*Límite de tasa"):
            OpenAICompatibleProvider(_FakeClient(response)).generate(PROMPT, config=_config())

    def test_network_error_raises_provider_error(self) -> None:
        class _Broken:
            def post(self, *args, **kwargs):
                raise TimeoutError("timeout")

        with self.assertRaisesRegex(GenerationProviderError, "No se pudo contactar"):
            OpenAICompatibleProvider(_Broken()).generate(PROMPT, config=_config())

    def test_missing_api_key_raises_configuration_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GenerationConfigurationError, "OPENROUTER_API_KEY"):
                OpenAICompatibleProvider(_FakeClient(None)).generate(PROMPT, config=_config())

    def test_generate_answer_selects_registered_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_generation_config(
                _write_config(Path(directory), {"provider": "openrouter", "model": MODEL})
            )
            client = _FakeClient(_FakeResponse(_completion()))
            with mock.patch.object(
                OpenAICompatibleProvider, "_create_client", return_value=client
            ):
                result = generate_answer(
                    ContextResult(query="q", context="c", chunks=[],
                                  metadata={"presentation": "documents", "expansion": "units"}),
                    config,
                )

        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(result.model, MODEL)
        self.assertEqual(result.answer, "Respuesta.")
        self.assertEqual(len(client.calls), 1)


class ChatCompletionParserTests(unittest.TestCase):
    def test_error_payload_raises(self) -> None:
        with self.assertRaisesRegex(GenerationProviderError, "Provider down"):
            parse_chat_completion({"error": {"message": "Provider down"}}, fallback_model="m")

    def test_missing_choices_raises(self) -> None:
        with self.assertRaisesRegex(GenerationProviderError, "sin choices"):
            parse_chat_completion(_completion(choices=[]), fallback_model="m")

    def test_empty_content_raises(self) -> None:
        data = _completion(
            choices=[{"finish_reason": "length", "message": {"content": ""}}]
        )
        with self.assertRaisesRegex(GenerationProviderError, "sin texto.*length"):
            parse_chat_completion(data, fallback_model="m")

    def test_missing_model_and_usage_use_fallbacks(self) -> None:
        data = _completion()
        del data["model"], data["usage"]
        result = parse_chat_completion(data, fallback_model="fallback")
        self.assertEqual(result.model, "fallback")
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)


class ProviderConfigTests(unittest.TestCase):
    def test_openrouter_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_generation_config(
                _write_config(Path(directory), {"provider": "OpenRouter", "model": MODEL})
            )
        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(config.extra_headers, {})
        self.assertEqual(config.extra_body, {})

    def test_gemini_config_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_generation_config(
                _write_config(Path(directory), {"provider": "gemini", "model": "g"})
            )
        self.assertEqual(config.api_key_env, "GEMINI_API_KEY")
        self.assertIsNone(config.base_url)

    def test_base_url_trailing_slash_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_generation_config(
                _write_config(
                    Path(directory),
                    {"provider": "openrouter", "model": MODEL, "base_url": "https://x.test/v1/"},
                )
            )
        self.assertEqual(config.base_url, "https://x.test/v1")

    def test_invalid_extra_fields_are_rejected(self) -> None:
        cases = [
            ({"extra_headers": {"X-Title": 1}}, TypeError),
            ({"extra_headers": []}, TypeError),
            ({"extra_body": "x"}, TypeError),
            ({"base_url": "openrouter.ai"}, ValueError),
        ]
        for extra, error in cases:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                path = _write_config(
                    Path(directory), {"provider": "openrouter", "model": MODEL, **extra}
                )
                with self.assertRaises(error):
                    load_generation_config(path)

    def test_registry(self) -> None:
        self.assertIsInstance(create_provider("gemini"), GeminiProvider)
        provider = create_provider("openrouter")
        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertEqual(provider.name, "openrouter")
        with self.assertRaisesRegex(GenerationConfigurationError, "Disponibles"):
            create_provider("desconocido")

    def test_example_config_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_generation_config(root / "config" / "generation.openrouter.example.json")
        self.assertEqual(config.model, MODEL)
        # Estrategia modular: system_prompt_path es la carpeta de prompts.
        self.assertEqual(config.prompt_strategy, "modular")
        self.assertTrue((config.system_prompt_path / "base_es.txt").is_file())


if __name__ == "__main__":
    unittest.main()

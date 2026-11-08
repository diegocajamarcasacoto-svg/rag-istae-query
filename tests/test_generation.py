import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from query.context.models import ContextChunk, ContextResult
from query.generation.errors import GenerationProviderError
from query.generation.generation_config import (
    GenerationConfig,
    load_generation_config,
)
from query.generation.generator import generate_answer
from query.generation.models import GenerationPrompt, ProviderResponse
from query.generation.providers.gemini import GeminiProvider
from query.generation.response_parser import parse_provider_response

MODULAR_FILES = ("base_es.txt", "rag_es.txt", "jerarquia_es.txt", "recordatorio_es.txt")


def _prompts_dir(root: Path) -> Path:
    prompts = root / "prompts"
    prompts.mkdir(exist_ok=True)
    for name in MODULAR_FILES:
        (prompts / name).write_text(f"Reglas {name}", encoding="utf-8")
    return prompts


def _context_result() -> ContextResult:
    return ContextResult(
        query="¿Qué dispone la norma?",
        context="=== [doc] Norma ===\n[doc-art18] Artículo 18.- El organismo deberá publicar el informe.",
        chunks=[
            ContextChunk(
                chunk_id="doc:18",
                document_id="doc",
                chunk_index=18,
                source_file="norma.pdf",
                page_numbers=[4],
                headings=["Artículo 18"],
                item_types=["article"],
                text="El organismo deberá publicar el informe.",
            )
        ],
        metadata={"seed_ids": ["doc:18"], "expansion": "units", "presentation": "documents"},
    )


def _config(prompt_path: Path) -> GenerationConfig:
    return GenerationConfig(
        provider="gemini",
        model="gemini-test",
        system_prompt_path=prompt_path,
        temperature=0.2,
        max_output_tokens=128,
        timeout_seconds=5,
        api_key_env="GEMINI_API_KEY",
    )


class _FakeProvider:
    name = "offline-fake"

    def __init__(self) -> None:
        self.prompt: GenerationPrompt | None = None

    def generate(
        self,
        prompt: GenerationPrompt,
        *,
        config: GenerationConfig,
    ) -> ProviderResponse:
        self.prompt = prompt
        return ProviderResponse(
            text="Debe publicar el informe.",
            model=config.model,
            input_tokens=25,
            output_tokens=7,
            finish_reason="STOP",
        )


class GenerationTests(unittest.TestCase):
    def test_load_generation_config_resolves_prompt_relative_to_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = _prompts_dir(root)
            config_path = root / "generation.json"
            config_path.write_text(
                json.dumps(
                    {
                        "provider": "gemini",
                        "model": "gemini-test",
                        "system_prompt_path": "prompts",
                    }
                ),
                encoding="utf-8",
            )

            config = load_generation_config(config_path)

        self.assertEqual(config.system_prompt_path, prompts.resolve())
        self.assertEqual(config.prompt_strategy, "modular")
        self.assertEqual(config.api_key_env, "GEMINI_API_KEY")
        self.assertEqual(config.max_output_tokens, 2048)

    def test_example_configs_use_modular_prompts(self) -> None:
        root = Path(__file__).resolve().parents[1] / "config"
        for name in ("generation.example.json", "generation.openrouter.example.json"):
            with self.subTest(name=name):
                config = load_generation_config(root / name)
                self.assertEqual(config.prompt_strategy, "modular")
                self.assertTrue((config.system_prompt_path / "base_es.txt").is_file())

    def test_gemini_provider_uses_generate_content_config_and_normalizes_response(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeModels:
            def generate_content(self, **kwargs: object) -> object:
                calls.append(kwargs)
                return SimpleNamespace(
                    text="Respuesta.",
                    model_version="gemini-test-version",
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=12,
                        candidates_token_count=4,
                    ),
                    candidates=[
                        SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))
                    ],
                )

        config = _config(Path("unused-prompt.txt"))
        prompt = GenerationPrompt("Reglas", '{"question":"Q","reference_context":"C"}')
        response = GeminiProvider(SimpleNamespace(models=FakeModels())).generate(
            prompt,
            config=config,
        )

        self.assertEqual(calls[0]["model"], "gemini-test")
        self.assertEqual(calls[0]["contents"], prompt.user_content)
        self.assertEqual(
            calls[0]["config"],
            {
                "system_instruction": "Reglas",
                "temperature": 0.2,
                "max_output_tokens": 128,
            },
        )
        self.assertEqual(response.text, "Respuesta.")
        self.assertEqual(response.model, "gemini-test-version")
        self.assertEqual(response.input_tokens, 12)
        self.assertEqual(response.output_tokens, 4)
        self.assertEqual(response.finish_reason, "STOP")

    def test_response_without_text_raises_provider_error(self) -> None:
        response = SimpleNamespace(
            text=None,
            candidates=[SimpleNamespace(finish_reason="SAFETY")],
            usage_metadata=None,
        )

        with self.assertRaisesRegex(GenerationProviderError, "sin texto"):
            parse_provider_response(response, fallback_model="model")

    def test_generation_preserves_context_trace_and_separates_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts = _prompts_dir(Path(directory))
            context_result = _context_result()
            fake_provider = _FakeProvider()

            result = generate_answer(
                context_result,
                _config(prompts),
                provider=fake_provider,
            )

        self.assertIs(result.context_result, context_result)
        self.assertEqual(result.context_result.metadata["seed_ids"], ["doc:18"])
        self.assertEqual(result.provider_metrics.input_tokens, 25)
        self.assertEqual(result.provider_metrics.output_tokens, 7)
        self.assertGreaterEqual(result.application_latency_ms, 0)
        # Prompt modular: documentos en el sistema; pregunta y recordatorio en el usuario.
        self.assertIn(f"<documentos>\n{context_result.context}\n</documentos>", fake_provider.prompt.system_instruction)
        self.assertTrue(fake_provider.prompt.user_content.startswith(f"<pregunta>\n{context_result.query}\n</pregunta>"))
        self.assertNotIn("seed_ids", fake_provider.prompt.user_content)
        self.assertEqual(result.to_dict()["context_result"], context_result.to_dict())

    def test_generation_requires_context_by_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts = _prompts_dir(Path(directory))
            context_result = _context_result()
            context_result.metadata = {"expansion": "neighbors"}
            with self.assertRaisesRegex(ValueError, "--expansion units"):
                generate_answer(context_result, _config(prompts), provider=_FakeProvider())


if __name__ == "__main__":
    unittest.main()

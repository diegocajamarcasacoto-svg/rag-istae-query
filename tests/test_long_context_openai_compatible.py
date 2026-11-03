"""Pruebas del adaptador compatible con OpenAI (OpenRouter) sin red."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.long_context.llm import ChatMessage, LLMRequest  # noqa: E402
from query.long_context.providers.openai_compatible import OpenAICompatibleClient, OpenAICompatibleError  # noqa: E402

REQUEST = LLMRequest(system="Instrucciones", context="[INT08-art12] texto", question=" ¿Pregunta? ",
                     history=(ChatMessage("user", "p1"), ChatMessage("assistant", "r1")), max_output_tokens=100)


def make(handler, **kwargs) -> tuple[OpenAICompatibleClient, list[dict]]:
    sent: list[dict] = []

    def wrapper(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content) if request.content else {})
        return handler(request)

    http = httpx.Client(base_url="https://x.test/api/v1", transport=httpx.MockTransport(wrapper))
    return OpenAICompatibleClient("nvidia/modelo:free", http_client=http, **kwargs), sent


class OpenAICompatibleTests(unittest.TestCase):
    def test_payload_keeps_documents_in_system_and_question_last(self) -> None:
        body = {"model": "nvidia/modelo", "provider": "Nvidia",
                "choices": [{"message": {"content": " Respuesta "}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 50,
                          "completion_tokens_details": {"reasoning_tokens": 20}, "cost": 0}}
        client, sent = make(lambda r: httpx.Response(200, json=body), reasoning_effort="medium")
        answer = client.generate(REQUEST)
        payload = sent[0]
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user", "assistant", "user"])
        self.assertIn("<documentos>", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][-1]["content"], "<pregunta>\n¿Pregunta?\n</pregunta>")
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["reasoning"], {"effort": "medium"})
        self.assertEqual(answer.text, "Respuesta")
        self.assertEqual((answer.input_tokens, answer.output_tokens), (900, 50))
        self.assertEqual(answer.extra["thoughts_tokens"], 20)
        self.assertFalse(answer.extra["model_mismatch"])

    def test_http_error_is_readable(self) -> None:
        client, _ = make(lambda r: httpx.Response(429, json={"error": {"message": "Rate limit", "metadata": {"raw": "free tier"}}}))
        with self.assertRaises(OpenAICompatibleError) as ctx:
            client.generate(REQUEST)
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("free tier", str(ctx.exception))

    def test_stream_parses_server_sent_events(self) -> None:
        events = ": OPENROUTER PROCESSING\n\n" + "".join(
            f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}\n\n" for t in ("Hola ", "mundo")
        ) + "data: [DONE]\n\n"
        client, sent = make(lambda r: httpx.Response(200, text=events))
        self.assertEqual("".join(client.stream(REQUEST)), "Hola mundo")
        self.assertTrue(sent[0]["stream"])


if __name__ == "__main__":
    unittest.main()

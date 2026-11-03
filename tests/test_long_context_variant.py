"""Pruebas de la variante de contexto largo con un proveedor simulado (sin red)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.long_context.catalog import RANK_LABELS, load_catalog  # noqa: E402
from query.long_context.citations import parse_answer  # noqa: E402
from query.long_context.corpus import load_corpus  # noqa: E402
from query.long_context.llm import ChatMessage, LLMRequest, LLMResponse  # noqa: E402
from query.long_context.variant import Answer, LongContextVariant  # noqa: E402

PROMPTS = ROOT / "config" / "prompts"
INT08 = """**Artículo 1.** Ámbito. Regula la titulación.

**Artículo 2.** Objeto. Normar el proceso.

**Artículo 12. Declaración de egresado.** Un estudiante será declarado egresado cuando cumpla con los siguientes requisitos:

- a) Declaratoria de aptitud con la documentación pertinente.

**Artículo 13. Requisitos para titularse.** Haber sido declarado egresado.
"""
EXT01 = "Art. 1.- Ámbito.- Esta Ley regula el sistema.\n\nArt. 2.- Objeto.- Definir principios.\n\nArt. 3.- Fines.- Formar profesionales."

GOOD_ANSWER = """Para ser declarado egresado necesitas la declaratoria de aptitud con la documentación pertinente [INT08-art12].

Fuentes:
- [INT08-art12] «Declaratoria de aptitud con la documentación pertinente»"""


class FakeClient:
    name = "fake"
    model = "fake-model-1"

    def __init__(self, text: str = GOOD_ANSWER) -> None:
        self.text = text
        self.requests: list[LLMRequest] = []

    def count_tokens(self, request: LLMRequest) -> int:
        self.requests.append(request)
        return len(request.system) + len(request.context) + len(request.question)

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=self.text, model=self.model, input_tokens=1000, cached_input_tokens=900, output_tokens=50, finish_reason="STOP")

    def stream(self, request: LLMRequest) -> Iterator[str]:
        self.requests.append(request)
        middle = len(self.text) // 2
        yield self.text[:middle]
        yield self.text[middle:]


class VariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        outputs = root / "outputs"
        (outputs / "extracted_text").mkdir(parents=True)
        rows = [
            {"document_id": "INT08", "relative_path": "INTERNOS/INT08 - TITULACION.docx", "source_sha256": "aaa"},
            {"document_id": "EXT01", "relative_path": "EXTERNOS/EXT01 - LOES.pdf", "source_sha256": "bbb"},
        ]
        (outputs / "document_mapping.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        (outputs / "extracted_text" / "aaa.txt").write_text(INT08, encoding="utf-8")
        (outputs / "extracted_text" / "bbb.txt").write_text(EXT01, encoding="utf-8")
        catalog = {
            "rank_labels": {str(k): v for k, v in RANK_LABELS.items()},
            "documents": [
                {"document_id": "EXT01", "code": "EXT-LOES", "name": "LOES", "doc_type": "Externo", "issuer": "Asamblea Nacional",
                 "domain": "", "rank": 2, "rank_label": RANK_LABELS[2], "source_file": "EXT01 - LOES.pdf"},
                {"document_id": "INT08", "code": "INT-RTG", "name": "Reglamento de Titulación", "doc_type": "Interno", "issuer": "OCS ISTAE",
                 "domain": "", "rank": 6, "rank_label": RANK_LABELS[6], "source_file": "INT08 - TITULACION.docx"},
            ],
            "relations": [],
        }
        path = root / "catalogo.json"
        path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
        catalog_docs, self.relations, meta = load_catalog(path)
        self.rank_labels = meta["rank_labels"]
        self.documents = load_corpus(outputs, catalog_docs)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make(self, variant: str = "A", client: FakeClient | None = None) -> LongContextVariant:
        return LongContextVariant(
            self.documents, variant, client or FakeClient(), PROMPTS, self.relations, self.rank_labels, max_history_messages=2
        )

    def test_prompts_name_the_right_institution_and_add_hierarchy_only_in_aj(self) -> None:
        plain, hierarchical = self.make("A"), self.make("AJ")
        self.assertIn("Instituto Superior Tecnológico Alberto Enríquez", plain.system)
        self.assertNotIn("JERARQUÍA NORMATIVA", plain.system)
        self.assertIn("JERARQUÍA NORMATIVA", hierarchical.system)
        self.assertNotEqual(plain.corpus_version, hierarchical.corpus_version)

    def test_request_keeps_question_last_and_limits_history(self) -> None:
        client = FakeClient()
        variant = self.make("A", client)
        history = [ChatMessage("user", "p1"), ChatMessage("assistant", "r1"), ChatMessage("user", "p2"), ChatMessage("assistant", "r2")]
        variant.answer("  ¿Qué necesito para egresar?  ", history)
        request = client.requests[-1]
        self.assertEqual(request.question, "¿Qué necesito para egresar?")
        self.assertEqual([m.content for m in request.history], ["p2", "r2"])
        self.assertEqual(request.temperature, 0.0)
        self.assertEqual(request.cache_key, variant.corpus_version)
        self.assertIn("[INT08-art12] ", request.context)

    def test_answer_verifies_citations_and_reports_usage(self) -> None:
        answer = self.make("A").answer("¿Qué necesito para egresar?")
        self.assertIsInstance(answer, Answer)
        self.assertEqual([c.unit_id for c in answer.citations], ["INT08-art12"])
        self.assertTrue(answer.citations[0].known)
        self.assertTrue(answer.citations[0].quote_verified)
        self.assertEqual(answer.verified_quote_ratio, 1.0)
        self.assertEqual(answer.cached_input_tokens, 900)
        self.assertFalse(answer.abstained)
        self.assertNotIn("Fuentes:", answer.body)

    def test_stream_yields_text_then_answer(self) -> None:
        events = list(self.make("AJ").stream("¿Qué necesito para egresar?"))
        self.assertEqual("".join(e for e in events if isinstance(e, str)), GOOD_ANSWER)
        self.assertIsInstance(events[-1], Answer)
        self.assertEqual(events[-1].variant, "AJ")

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.make().answer("   ")


class CitationTests(unittest.TestCase):
    UNITS = {"INT08-art12": ("INT08", "**Artículo 12. Declaración de egresado.** Un estudiante será declarado egresado cuando cumpla: a) Declaratoria de aptitud con la documentación pertinente.")}

    def test_detects_invented_id_and_unverified_quote(self) -> None:
        text = "Requisito A [INT08-art12] y requisito B [INT08-art99].\n\nFuentes:\n- [INT08-art12] «aprobar un examen de suficiencia final»\n- [INT08-art99] «texto»"
        parsed = parse_answer(text, self.UNITS)
        self.assertEqual(parsed.unknown_ids, ["INT08-art99"])
        self.assertFalse(parsed.citations[0].quote_verified)

    def test_quote_matching_ignores_accents_format_and_ellipsis(self) -> None:
        text = "X [INT08-art12].\n\n**Fuentes:**\n- [INT08-art12] «Un estudiante sera declarado egresado ... declaratoria de aptitud»"
        parsed = parse_answer(text, self.UNITS)
        self.assertTrue(parsed.citations[0].quote_verified)

    def test_quote_matches_across_list_bullets(self) -> None:
        units = {"INT08-art12": ("INT08", "Requisitos:\n\n- a) Declaratoria de aptitud.\n- b) Nivel A1 en técnico.")}
        text = "X [INT08-art12].\n\nFuentes:\n- [INT08-art12] «Requisitos: a) Declaratoria de aptitud. b) Nivel A1 en técnico.»"
        self.assertTrue(parse_answer(text, units).citations[0].quote_verified)

    def test_quote_keeps_inline_list_number_from_source(self) -> None:
        units = {"INT08-art75": ("INT08", "Podrá solicitar cambio de opción:\n\n1. Por una sola ocasión por fuerza mayor.")}
        text = "X [INT08-art75].\n\nFuentes:\n- [INT08-art75] «Podrá solicitar cambio de opción: 1. Por una sola ocasión por fuerza mayor.»"
        self.assertTrue(parse_answer(text, units).citations[0].quote_verified)

    def test_detects_abstention_and_citations_without_source(self) -> None:
        self.assertTrue(parse_answer("No encontré en la normativa disponible información sobre becas.", self.UNITS).abstained)
        parsed = parse_answer("Algo [INT08-art12].", self.UNITS)
        self.assertEqual(parsed.cited_without_source, ["INT08-art12"])
        self.assertIsNone(parsed.verified_quote_ratio)


if __name__ == "__main__":
    unittest.main()


class QuoteStartingWithLiteralTests(unittest.TestCase):
    def test_quote_that_opens_with_a_literal(self) -> None:
        units = {"INT08-art11": ("INT08", "Documentos:\n\n- a) Certificados de matrículas.\n- b) Record académico de la carrera.")}
        text = "X [INT08-art11].\n\nFuentes:\n- [INT08-art11] «a) Certificados de matrículas. b) Record académico de la carrera.»"
        self.assertTrue(parse_answer(text, units).citations[0].quote_verified)

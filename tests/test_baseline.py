"""Fija los parámetros de la línea base B/C (LB-1, 2026-11-06).

Si una de estas pruebas falla, se cambió un valor por defecto de la línea base:
hágalo solo como decisión explícita y actualice la sección «Línea base B/C» del README.
"""

import unittest
from pathlib import Path

import query_rag

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "pregunta"


def _args(*extra: str):
    parser = query_rag._parser()
    args = parser.parse_args([QUESTION, *extra])
    query_rag._validate_args(parser, args)
    return args


class BaselineTests(unittest.TestCase):
    def test_baseline_b_retrieval(self) -> None:
        args = _args("--expansion", "units")
        self.assertEqual(args.retrieval_mode, "hybrid")
        self.assertEqual((args.top_k, args.candidate_k, args.prefetch_k, args.rrf_k), (5, 20, 20, 60))
        self.assertTrue(args.rerank)
        self.assertIsNone(args.max_context_tokens)  # → DEFAULT_UNITS_MAX_CONTEXT_TOKENS
        self.assertEqual(query_rag.DEFAULT_UNITS_MAX_CONTEXT_TOKENS, 4000)
        self.assertEqual(args.max_unit_tokens, 1500)

    def test_baseline_c_hierarchy(self) -> None:
        args = _args("--retrieval-mode", "hierarchical")
        self.assertEqual(args.expansion, "units")
        self.assertEqual(args.hierarchy_relations, "Desarrolla,Debe ser compatible con")
        self.assertEqual(args.hierarchy_depth, 2)
        self.assertEqual(args.hierarchy_search, "joint")
        self.assertEqual(args.hierarchy_candidate_k, 20)
        self.assertEqual((args.hierarchy_max_units, args.hierarchy_max_per_document), (3, 1))
        self.assertEqual(args.hierarchy_max_tokens, 1500)
        self.assertIsNone(args.hierarchy_min_score)

    def test_baseline_prompts(self) -> None:
        modular = ROOT / "config" / "prompts" / "modular"
        for name in ("base_es.txt", "rag_es.txt", "jerarquia_es.txt", "recordatorio_es.txt"):
            self.assertTrue((modular / name).is_file(), name)
        rag = (modular / "rag_es.txt").read_text(encoding="utf-8")
        # La frase de abstención se reserva para cuando no hay información pertinente.
        self.assertIn('Empieza con "No encontré en la normativa disponible" solo si ninguno', rag)
        self.assertIn("no empieces con esa frase", rag)


if __name__ == "__main__":
    unittest.main()

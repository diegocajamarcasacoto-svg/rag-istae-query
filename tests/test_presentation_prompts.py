"""Pruebas de la presentación por documentos y del prompt modular (formato rag-istae-api)."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from query.context.models import ContextResult
from query.generation.generation_config import load_generation_config
from query.generation.prompt_builder import build_modular_prompt, load_modular_prompts
from query.hierarchy import RELATION_COMPATIBLE, RELATION_DEVELOPS, catalog_from_dict, expand_with_hierarchy
from query.presentation import INCLUDED_BY_HIERARCHY, render_documents
from query.units import Unit, UnitIndex, build_unit_context

ROOT = Path(__file__).resolve().parents[1]
MODULAR_DIR = ROOT / "config" / "prompts" / "modular"

CATALOG = catalog_from_dict({
    "rank_labels": {"2": "Ley orgánica", "5": "Estatuto", "6": "Reglamento interno"},
    "documents": [
        {"document_id": "EXT01", "name": "LOES", "rank": 2, "rank_label": "Ley orgánica", "issuer": "Asamblea"},
        {"document_id": "INT01", "name": "Estatuto", "rank": 5, "rank_label": "Estatuto"},
        {"document_id": "INT08", "name": "Reglamento de Titulación", "rank": 6, "rank_label": "Reglamento interno"},
        {"document_id": "INT12", "name": "Inasistencia", "rank": 6, "rank_label": "Reglamento interno"},
    ],
    "relations": [
        {"source_document_id": "INT08", "relation": RELATION_DEVELOPS, "target_document_id": "INT01"},
        {"source_document_id": "INT01", "relation": RELATION_DEVELOPS, "target_document_id": "EXT01"},
        {"source_document_id": "INT12", "relation": RELATION_COMPATIBLE, "target_document_id": "EXT01"},
    ],
})


def _point(chunk_id: str, score: float = 0.5) -> SimpleNamespace:
    document_id = chunk_id.split(":")[0]
    return SimpleNamespace(payload={"chunk_id": chunk_id, "document_id": document_id, "text": chunk_id}, score=score)


def _index() -> UnitIndex:
    return UnitIndex([
        Unit("INT08-art12", "INT08", "articulo", "Art. 12", "Egresado", "", ("INT08:00039",), "Artículo 12. Egresado."),
        Unit("INT08-art13", "INT08", "articulo", "Art. 13", "", "", ("INT08:00042",), "Artículo 13. Titulación."),
        Unit("INT12-art2", "INT12", "articulo", "Art. 2", "", "", ("INT12:00022",), "Artículo 2. Ámbito."),
        Unit("INT01-art5", "INT01", "articulo", "Art. 5", "", "", ("INT01:00010",), "Artículo 5. Fines."),
    ])


def _hybrid() -> ContextResult:
    ranked = [(_point("INT08:00039"), 0.9), (_point("INT12:00022"), 0.8), (_point("INT08:00042"), 0.7)]
    return build_unit_context("pregunta", ranked, _index())


def _hierarchical() -> ContextResult:
    base = build_unit_context("pregunta", [(_point("INT08:00039"), 0.9), (_point("INT12:00022"), 0.8)], _index())
    return expand_with_hierarchy("pregunta", base, _index(), CATALOG,
                                 lambda ids: [(_point("INT01:00010"), 0.6)],
                                 relations=(RELATION_DEVELOPS,), max_depth=1)


class RenderDocumentsTests(unittest.TestCase):
    def test_hybrid_groups_units_by_document_in_relevance_order(self) -> None:
        result = render_documents(_hybrid(), CATALOG)
        self.assertEqual(
            result.context,
            "=== [INT08] Reglamento de Titulación ===\n"
            "[INT08-art12] Artículo 12. Egresado.\n\n"
            "[INT08-art13] Artículo 13. Titulación.\n\n"
            "=== [INT12] Inasistencia ===\n"
            "[INT12-art2] Artículo 2. Ámbito.",
        )
        self.assertEqual(result.metadata["presentation"], "documents")
        self.assertEqual(result.metadata["document_order"], ["INT08", "INT12"])
        self.assertEqual(result.metadata["metrics"]["token_count"], -(-len(result.context.encode()) // 4))
        self.assertEqual([c.chunk_id for c in result.chunks], ["INT08-art12", "INT12-art2", "INT08-art13"])

    def test_hybrid_without_catalog_uses_document_id(self) -> None:
        result = render_documents(_hybrid(), None)
        self.assertTrue(result.context.startswith("=== [INT08] INT08 ===\n[INT08-art12]"))

    def test_hierarchical_like_aj_with_marker_and_present_relations(self) -> None:
        result = render_documents(_hierarchical(), CATALOG)
        context = result.context
        self.assertTrue(context.startswith("=== ESCALA DE JERARQUÍA NORMATIVA (1 = mayor rango) ===\n2. Ley orgánica"))
        self.assertEqual(result.metadata["document_order"], ["INT01", "INT08", "INT12"])  # rango, luego relevancia
        self.assertIn(f"=== [INT01] Estatuto · Jerarquía 5: Estatuto · Emisor: no registrado · {INCLUDED_BY_HIERARCHY} ===\n"
                      "[INT01-art5] Artículo 5. Fines.", context)
        self.assertIn("=== [INT08] Reglamento de Titulación · Jerarquía 6: Reglamento interno · Emisor: no registrado ===\n"
                      "[INT08-art12] Artículo 12. Egresado.", context)
        relations = context.split("=== RELACIONES ENTRE NORMAS ===\n")[1]
        self.assertEqual(relations, "- [INT08] Reglamento de Titulación — Desarrolla → [INT01] Estatuto")  # EXT01 ausente
        self.assertNotIn("NORMAS DE RANGO SUPERIOR", context)

    def test_requires_units_and_catalog_for_hierarchical(self) -> None:
        with self.assertRaises(ValueError):
            render_documents(ContextResult("q", "x", [], {"expansion": "neighbors"}), CATALOG)
        with self.assertRaises(ValueError):
            render_documents(_hierarchical(), None)


class ModularPromptTests(unittest.TestCase):
    def test_hybrid_uses_base_and_rag_only(self) -> None:
        prompt = build_modular_prompt(render_documents(_hybrid(), CATALOG), MODULAR_DIR)
        system = prompt.system_instruction
        self.assertTrue(system.startswith((MODULAR_DIR / "base_es.txt").read_text(encoding="utf-8").strip()))
        self.assertIn("DOCUMENTOS RECUPERADOS", system)
        self.assertNotIn("JERARQUÍA NORMATIVA\n", system)
        self.assertTrue(system.endswith("<documentos>\n=== [INT08] Reglamento de Titulación ===\n"
                                        "[INT08-art12] Artículo 12. Egresado.\n\n[INT08-art13] Artículo 13. Titulación.\n\n"
                                        "=== [INT12] Inasistencia ===\n[INT12-art2] Artículo 2. Ámbito.\n</documentos>"))
        reminder = (MODULAR_DIR / "recordatorio_es.txt").read_text(encoding="utf-8").strip()
        self.assertEqual(prompt.user_content, f"<pregunta>\npregunta\n</pregunta>\n\n{reminder}")

    def test_hierarchical_adds_hierarchy_rules(self) -> None:
        system, _ = load_modular_prompts(MODULAR_DIR, hierarchical=True)
        self.assertLess(system.index("DOCUMENTOS RECUPERADOS"), system.index("JERARQUÍA NORMATIVA"))
        prompt = build_modular_prompt(render_documents(_hierarchical(), CATALOG), MODULAR_DIR)
        self.assertIn("JERARQUÍA NORMATIVA", prompt.system_instruction)

    def test_requires_presentation_by_documents(self) -> None:
        with self.assertRaises(ValueError):
            build_modular_prompt(_hybrid(), MODULAR_DIR)

    def test_hierarchy_rules_do_not_invoke_norms_outside_the_documents(self) -> None:
        text = (MODULAR_DIR / "jerarquia_es.txt").read_text(encoding="utf-8")
        self.assertNotIn("425", text)
        self.assertNotIn("Constitución", text)
        # La abstención conserva la frase que detecta rag-istae-api.
        self.assertIn("No encontré en la normativa disponible", (MODULAR_DIR / "rag_es.txt").read_text(encoding="utf-8"))


class PromptStrategyConfigTests(unittest.TestCase):
    def _write(self, directory: str, **data) -> Path:
        path = Path(directory) / "generation.json"
        path.write_text(json.dumps({"provider": "openrouter", "model": "m", **data}), encoding="utf-8")
        return path

    def test_modular_strategy_resolves_prompt_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_generation_config(self._write(directory, system_prompt_path="prompts/modular",
                                                        prompt_strategy="modular"))
            self.assertEqual(config.prompt_strategy, "modular")
            self.assertEqual(config.system_prompt_path, (Path(directory) / "prompts" / "modular").resolve())

    def test_default_and_retired_strategies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                load_generation_config(self._write(directory, system_prompt_path="prompts/modular")).prompt_strategy,
                "modular",
            )
            # La estrategia "single" se retiró, y un archivo en lugar de la carpeta es un error.
            with self.assertRaisesRegex(ValueError, "single"):
                load_generation_config(self._write(directory, system_prompt_path="prompts/modular",
                                                   prompt_strategy="single"))
            (Path(directory) / "s.txt").write_text("Reglas", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "carpeta"):
                load_generation_config(self._write(directory, system_prompt_path="s.txt"))


if __name__ == "__main__":
    unittest.main()

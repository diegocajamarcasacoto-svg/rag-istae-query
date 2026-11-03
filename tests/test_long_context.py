"""Pruebas del corpus de contexto largo con textos de ejemplo (sin red ni modelos)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from query.long_context.catalog import RANK_LABELS, load_catalog, propose_rank  # noqa: E402
from query.long_context.cleaning import clean_markdown  # noqa: E402
from query.long_context.corpus import load_corpus, render_corpus, units_records  # noqa: E402
from query.long_context.structure import parse_document  # noqa: E402

ARTICULATED = """**REGLAMENTO DE TITULACIÓN**

Considerando:

Que, el artículo 3 de la Ley establece fines.

**CAPÍTULO I**

**Artículo 1** . Ámbito. El presente reglamento regula la titulación.

**Artículo 2.** Objetivos. Los objetivos son:

- a) Normar el proceso.
- b) Garantizar la calidad.

Art. 95.- Criterios de otra ley citados textualmente.

**CAPÍTULO II**

**Artículo 3. Declaración de egresado.** Un estudiante será declarado egresado cuando:

- a) Apruebe el plan de estudios.

**Artículo 3. Duplicado por error del documento.** Texto.

**DISPOSICIONES TRANSITORIAS**

**PRIMERA.-** Los procesos en curso concluirán con la norma anterior.

**SEGUNDA.-** Esta norma entra en vigencia al aprobarse.
"""

SECTIONS = """**NORMATIVA PARA LOS TRABAJOS DE INTEGRACIÓN CURRICULAR**

**OBJETIVO**

Regular el proceso de presentación de trabajos.

**ALCANCE**

**ANTEPROYECTO**

- Durante el tercer periodo el estudiante presentará el anteproyecto.

Art. 87.- Texto citado de la LOES.
"""


class StructureTests(unittest.TestCase):
    def test_articulated_ids_are_stable_and_follow_numbering(self) -> None:
        parsed = parse_document("INT08", ARTICULATED)
        ids = [unit.unit_id for unit in parsed.units]
        self.assertEqual(parsed.mode, "articulado")
        self.assertEqual(ids[0], "INT08-preambulo")
        self.assertIn("INT08-art1", ids)
        self.assertIn("INT08-art3", ids)
        self.assertIn("INT08-art3-2", ids)  # duplicado del documento original
        self.assertIn("INT08-disp-transitoria-primera", ids)
        self.assertIn("INT08-disp-transitoria-segunda", ids)

    def test_out_of_sequence_article_is_kept_as_text_not_as_unit(self) -> None:
        parsed = parse_document("INT08", ARTICULATED)
        ids = [unit.unit_id for unit in parsed.units]
        self.assertNotIn("INT08-art95", ids)
        art2 = next(unit for unit in parsed.units if unit.unit_id == "INT08-art2")
        self.assertIn("Art. 95.- Criterios", art2.text)
        self.assertEqual(len(parsed.rejected_article_candidates), 1)

    def test_article_keeps_its_list_and_path(self) -> None:
        parsed = parse_document("INT08", ARTICULATED)
        art3 = next(unit for unit in parsed.units if unit.unit_id == "INT08-art3")
        self.assertIn("a) Apruebe el plan de estudios", art3.text)
        self.assertEqual(art3.title, "Declaración de egresado")
        self.assertEqual(art3.path, "CAPÍTULO II")
        self.assertTrue(art3.text.startswith("**CAPÍTULO II**"))

    def test_text_is_preserved_in_order(self) -> None:
        parsed = parse_document("INT08", ARTICULATED)
        rebuilt = "\n".join(unit.text for unit in parsed.units)
        normalize = lambda value: [line for line in value.split("\n") if line.strip()]  # noqa: E731
        self.assertEqual(normalize(rebuilt), normalize(ARTICULATED))

    def test_document_without_articles_uses_sections(self) -> None:
        parsed = parse_document("INT09", SECTIONS)
        self.assertEqual(parsed.mode, "secciones")
        titles = {unit.unit_id: unit.title for unit in parsed.units}
        # Encabezados consecutivos sin texto entre ellos se fusionan en una sola sección.
        self.assertEqual(titles["INT09-sec01"], "NORMATIVA PARA LOS TRABAJOS DE INTEGRACIÓN CURRICULAR › OBJETIVO")
        self.assertEqual(titles["INT09-sec02"], "ALCANCE › ANTEPROYECTO")
        self.assertNotIn("INT09-art87", titles)


class NumberingGapTests(unittest.TestCase):
    def test_confirmed_jump_is_accepted_and_reported(self) -> None:
        text = "Art. 1.- Uno.\nArt. 2.- Dos.\nArt. 3.- Tres.\nArt. 9.- Nueve.\nArt. 10.- Diez."
        parsed = parse_document("EXT02", text)
        ids = [unit.unit_id for unit in parsed.units]
        self.assertIn("EXT02-art9", ids)
        self.assertIn("EXT02-art10", ids)
        self.assertTrue(any("3 → 9" in warning for warning in parsed.warnings))

    def test_isolated_quote_is_rejected(self) -> None:
        text = "Art. 1.- Uno.\nArt. 2.- Dos.\nArt. 95.- Cita.\nArt. 3.- Tres."
        parsed = parse_document("EXT02", text)
        ids = [unit.unit_id for unit in parsed.units]
        self.assertNotIn("EXT02-art95", ids)
        self.assertIn("EXT02-art3", ids)


class CleaningTests(unittest.TestCase):
    def test_removes_noise_but_keeps_first_letterhead_and_content(self) -> None:
        text = "UBICACIÓN: San Lorenzo\nArtículo 1. Texto\n<!-- image -->\n14\nUBICACIÓN: San Lorenzo\nTipo de indicador: Cualitativo\nTipo de indicador: Cualitativo"
        cleaned, report = clean_markdown(text)
        self.assertEqual(cleaned.count("UBICACIÓN: San Lorenzo"), 1)
        self.assertEqual(cleaned.count("Tipo de indicador: Cualitativo"), 2)
        self.assertNotIn("<!-- image -->", cleaned)
        self.assertEqual(report.image_placeholders, 1)
        self.assertEqual(report.page_numbers, 1)
        self.assertEqual(report.removed_lines, 3)


class CatalogTests(unittest.TestCase):
    def test_rank_proposal_rules(self) -> None:
        self.assertEqual(propose_rank("Externo", "Asamblea Nacional", "Ley Orgánica de Educación Superior (LOES)", "EXT-LOES"), 2)
        self.assertEqual(propose_rank("Externo", "CES", "Reglamento de Régimen Académico", "EXT-RRA"), 3)
        self.assertEqual(propose_rank("Externo", "CACES", "Modelo de Evaluación Externa 2024", "EXT-MEE24"), 4)
        self.assertEqual(propose_rank("Interno", "OCS ISTAE", "Estatuto Interno ISTAE-2025", "INT-EST"), 5)
        self.assertEqual(propose_rank("Interno", "OCS ISTAE", "Reglamento de Titulación", "INT-RTG"), 6)
        self.assertEqual(set(RANK_LABELS), {1, 2, 3, 4, 5, 6})


class CorpusTests(unittest.TestCase):
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
        (outputs / "extracted_text" / "aaa.txt").write_text(ARTICULATED, encoding="utf-8")
        (outputs / "extracted_text" / "bbb.txt").write_text(
            "Art. 1.- Ámbito.- Esta Ley regula.\n\nArt. 2.- Objeto.- Definir.\n\nArt. 3.- Fines.- Formar.", encoding="utf-8"
        )
        catalog = {
            "rank_labels": {str(k): v for k, v in RANK_LABELS.items()},
            "rank_status": "propuesta",
            "documents": [
                {"document_id": "EXT01", "code": "EXT-LOES", "name": "LOES", "doc_type": "Externo", "issuer": "Asamblea Nacional",
                 "domain": "", "rank": 2, "rank_label": RANK_LABELS[2], "source_file": "EXT01 - LOES.pdf"},
                {"document_id": "INT08", "code": "INT-RTG", "name": "Reglamento de Titulación", "doc_type": "Interno", "issuer": "OCS ISTAE",
                 "domain": "", "rank": 6, "rank_label": RANK_LABELS[6], "source_file": "INT08 - TITULACION.docx"},
            ],
            "relations": [
                {"source_document_id": "INT08", "source_code": "INT-RTG", "relation": "Desarrolla",
                 "target_document_id": "EXT01", "target_code": "EXT-LOES", "verification": ""},
            ],
        }
        self.catalog_path = root / "catalogo.json"
        self.catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
        self.outputs = outputs

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_variants_share_text_and_differ_in_order_and_hierarchy(self) -> None:
        catalog, relations, meta = load_catalog(self.catalog_path)
        documents = load_corpus(self.outputs, catalog)
        corpus_a = render_corpus(documents, "A")
        corpus_aj = render_corpus(documents, "AJ", relations, meta["rank_labels"])

        self.assertLess(corpus_a.index("[EXT01]"), corpus_a.index("[INT08]"))
        self.assertIn("[INT08-art3] ", corpus_a)
        self.assertNotIn("Jerarquía", corpus_a)
        self.assertNotIn("RELACIONES ENTRE NORMAS", corpus_a)

        self.assertTrue(corpus_aj.startswith("=== ESCALA DE JERARQUÍA NORMATIVA"))
        self.assertIn("Jerarquía 2: Ley orgánica", corpus_aj)
        self.assertIn("- [INT08] Reglamento de Titulación — Desarrolla → [EXT01] LOES", corpus_aj)
        for unit in units_records(documents):
            self.assertIn(f"[{unit['unit_id']}] ", corpus_aj)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.context.structure_detector import detect_structure


class StructureDetectorTests(unittest.TestCase):
    def test_article_in_main_text_overrides_auxiliary_concordance_heading(self):
        result = detect_structure(
            {
                "headings": ["Concordancias:"],
                "text": "Artículo 18.- El contenido principal establece los requisitos.",
            }
        )
        self.assertEqual(result["type"], "article")
        self.assertEqual(result["id"], "18")

    def test_explicit_concordance_marker_in_main_text_is_detected(self):
        result = detect_structure(
            {
                "headings": [],
                "text": "Concordancias:\nCódigo Civil, Arts. 12 y 15.",
            }
        )
        self.assertEqual(result["type"], "concordance")

    def test_article_can_be_identified_from_heading_without_conflicting_structure(self):
        result = detect_structure(
            {
                "headings": ["Artículo 27.- Derechos de los estudiantes"],
                "text": "Se reconocen los siguientes derechos a los estudiantes.",
            }
        )
        self.assertEqual(result["type"], "article")
        self.assertEqual(result["id"], "27")

    def test_concordance_heading_without_marker_does_not_override_other_signals(self):
        result = detect_structure(
            {
                "headings": ["Concordancias:"],
                "text": "Artículo 19.- El contenido principal corresponde a este artículo.",
            }
        )
        self.assertEqual(result["type"], "article")
        self.assertEqual(result["id"], "19")

    def test_concordance_heading_with_normative_references_is_unknown(self):
        result = detect_structure(
            {
                "headings": ["Concordancias:"],
                "text": "Constitución, Art. 10; Ley Orgánica, Arts. 3 y 7.",
            }
        )
        self.assertEqual(result["type"], "unknown")
        self.assertEqual(result["status"], "unknown")

    def test_content_without_recognizable_structure_is_unknown(self):
        result = detect_structure(
            {
                "headings": [],
                "text": "Explicación general sin artículo, concordancia ni reforma.",
                "item_types": ["text"],
            }
        )
        self.assertEqual(result["type"], "unknown")
        self.assertEqual(result["status"], "unknown")


if __name__ == "__main__":
    unittest.main()

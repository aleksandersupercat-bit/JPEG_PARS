import sys
import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jpeg_pars.template_parser import (
    OcrCandidate,
    ParsedSheet,
    TemplateRegion,
    _contains_plus_minus_symbol,
    _merge_plus_minus_tokens,
    _psm_candidates,
    default_region_name,
    export_results_to_excel,
    load_template,
    normalize_ocr_text,
    save_template,
)


class TemplateParserTests(unittest.TestCase):
    def test_default_region_name_uses_alphabet(self) -> None:
        self.assertEqual(default_region_name(0), "A")
        self.assertEqual(default_region_name(23), "X")
        self.assertEqual(default_region_name(24), "A1")

    def test_export_results_to_excel_writes_headers_and_values(self) -> None:
        regions = [
            TemplateRegion(name="A", color="#FF0000", x0=0.1, y0=0.1, x1=0.2, y1=0.2),
            TemplateRegion(name="B", color="#00FF00", x0=0.3, y0=0.3, x1=0.4, y1=0.4),
        ]
        rows = [
            ParsedSheet(
                file_name="sheet1.jpg",
                file_path="C:/data/sheet1.jpg",
                values={"A": "120", "B": "45%"},
                confidences={"A": 92.5, "B": 81.0},
                debug_candidates={
                    "A": [OcrCandidate("gray_up", "paddle", "120", 92.5, 95.0)],
                    "B": [OcrCandidate("otsu", "paddle", "45%", 81.0, 83.0)],
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / "parsed.xlsx"
            export_results_to_excel(rows, regions, destination)

            workbook = load_workbook(destination)
            sheet = workbook.active
            self.assertEqual(sheet.cell(row=1, column=1).value, "file_name")
            self.assertEqual(sheet.cell(row=1, column=3).value, "A")
            self.assertEqual(sheet.cell(row=1, column=4).value, "A_confidence")
            self.assertEqual(sheet.cell(row=2, column=1).value, "sheet1.jpg")
            self.assertEqual(sheet.cell(row=2, column=5).value, "45%")
            self.assertEqual(sheet.cell(row=2, column=6).value, 81.0)

    def test_save_and_load_template_roundtrip(self) -> None:
        regions = [
            TemplateRegion(name="A", color="#FF0000", x0=0.1, y0=0.1, x1=0.2, y1=0.2),
            TemplateRegion(name="B", color="#00FF00", x0=0.3, y0=0.3, x1=0.4, y1=0.4),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / "template.json"
            image_path = Path(tmp_dir) / "source.jpg"
            image_path.write_text("stub", encoding="utf-8")
            save_template(regions, image_path, destination)
            loaded_image_path, loaded_regions = load_template(destination)

            self.assertEqual(loaded_image_path, image_path)
            self.assertEqual(len(loaded_regions), 2)
            self.assertEqual(loaded_regions[1].name, "B")

    def test_normalize_ocr_text_preserves_plus_minus_and_percent(self) -> None:
        self.assertEqual(normalize_ocr_text("10 +- 0.2"), "10±0.2")
        self.assertEqual(normalize_ocr_text("25 + / - 1"), "25±1")
        self.assertEqual(normalize_ocr_text("45 %"), "45%")

    def test_psm_candidates_include_vertical_friendly_modes(self) -> None:
        candidates = _psm_candidates(6)
        self.assertIn(5, candidates)
        self.assertIn(11, candidates)
        self.assertIn(13, candidates)

    def test_merge_plus_minus_tokens_rebuilds_measurement(self) -> None:
        self.assertEqual(_merge_plus_minus_tokens("434+5"), "434±5")
        self.assertEqual(_merge_plus_minus_tokens("434 5"), "434±5")
        self.assertEqual(_merge_plus_minus_tokens("434=5"), "434±5")

    def test_contains_plus_minus_symbol_detects_simple_glyph(self) -> None:
        image = Image.new("L", (30, 30), color=255)
        draw = ImageDraw.Draw(image)
        draw.line((10, 6, 20, 6), fill=0, width=2)
        draw.line((10, 14, 20, 14), fill=0, width=2)
        draw.line((10, 22, 20, 22), fill=0, width=2)
        draw.line((15, 10, 15, 18), fill=0, width=2)
        self.assertTrue(_contains_plus_minus_symbol(image))

if __name__ == "__main__":
    unittest.main()

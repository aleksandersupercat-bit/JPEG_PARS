import tempfile
import unittest
from pathlib import Path
import sys

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jpeg_pars.template_parser import (
    ParsedSheet,
    TemplateRegion,
    default_region_name,
    export_results_to_excel,
    load_template,
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
                values={"A": "120", "B": "450"},
                confidences={"A": 92.5, "B": 81.0},
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
            self.assertEqual(sheet.cell(row=2, column=5).value, "450")
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


if __name__ == "__main__":
    unittest.main()

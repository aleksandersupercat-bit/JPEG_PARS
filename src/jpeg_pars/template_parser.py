from __future__ import annotations

import json
import shutil
import string
from dataclasses import asdict, dataclass
from pathlib import Path

from openpyxl import Workbook
from PIL import Image, ImageOps

from .features import OcrConfig, SUPPORTED_EXTENSIONS

try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
except ImportError:
    pytesseract = None
    TesseractOutput = None


@dataclass(slots=True)
class TemplateRegion:
    name: str
    color: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(slots=True)
class ParsedSheet:
    file_name: str
    file_path: str
    values: dict[str, str]
    confidences: dict[str, float]


@dataclass(slots=True)
class OcrExtraction:
    text: str
    confidence: float


def list_jpeg_files(folder: Path, recursive: bool = False) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def default_region_name(index: int) -> str:
    alphabet = string.ascii_uppercase[:24]
    if index < len(alphabet):
        return alphabet[index]
    prefix = alphabet[index % len(alphabet)]
    suffix = index // len(alphabet)
    return f"{prefix}{suffix}"


def parse_template_batch(
    image_folder: Path,
    regions: list[TemplateRegion],
    ocr_config: OcrConfig,
    recursive: bool = False,
) -> list[ParsedSheet]:
    _ensure_ocr_ready(ocr_config)
    files = list_jpeg_files(image_folder, recursive=recursive)
    results: list[ParsedSheet] = []
    for path in files:
        image = Image.open(path).convert("RGB")
        values: dict[str, str] = {}
        confidences: dict[str, float] = {}
        for region in regions:
            extraction = extract_region_text(image, region, ocr_config)
            values[region.name] = extraction.text
            confidences[region.name] = extraction.confidence
        results.append(
            ParsedSheet(
                file_name=path.name,
                file_path=str(path),
                values=values,
                confidences=confidences,
            )
        )
    return results


def extract_region_text(image: Image.Image, region: TemplateRegion, ocr_config: OcrConfig) -> OcrExtraction:
    _apply_tesseract_cmd(ocr_config)
    width, height = image.size
    box = (
        int(region.x0 * width),
        int(region.y0 * height),
        int(region.x1 * width),
        int(region.y1 * height),
    )
    cropped = image.crop(box)
    if cropped.width <= 2 or cropped.height <= 2:
        return OcrExtraction(text="", confidence=0.0)

    variants = _ocr_variants(cropped)
    best = OcrExtraction(text="", confidence=0.0)
    best_score = -1.0
    for variant in variants:
        extraction = _extract_text_with_confidence(variant, ocr_config)
        score = _ocr_text_score(extraction.text) + extraction.confidence * 0.2
        if score > best_score:
            best_score = score
            best = extraction
    return best


def export_results_to_excel(
    rows: list[ParsedSheet],
    regions: list[TemplateRegion],
    destination: Path,
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Parsed"
    headers = ["file_name", "file_path"]
    for region in regions:
        headers.append(region.name)
        headers.append(f"{region.name}_confidence")
    sheet.append(headers)

    for row in rows:
        values = [row.file_name, row.file_path]
        for region in regions:
            values.append(row.values.get(region.name, ""))
            values.append(round(row.confidences.get(region.name, 0.0), 2))
        sheet.append(values)

    for column_cells in sheet.columns:
        max_length = max(len(str(cell.value or "")) for cell in column_cells)
        sheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 40)

    destination.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(destination)


def save_template(regions: list[TemplateRegion], image_path: Path | None, destination: Path) -> None:
    payload = {
        "image_path": (str(image_path) if image_path else None),
        "regions": [asdict(region) for region in regions],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_template(source: Path) -> tuple[Path | None, list[TemplateRegion]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    image_path = Path(payload["image_path"]) if payload.get("image_path") else None
    regions = [TemplateRegion(**item) for item in payload.get("regions", [])]
    return image_path, regions


def normalize_ocr_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").split()).strip()


def _ocr_variants(image: Image.Image) -> list[Image.Image]:
    grayscale = ImageOps.autocontrast(image.convert("L"))
    grayscale = grayscale.resize((max(1, grayscale.width * 3), max(1, grayscale.height * 3)), Image.Resampling.LANCZOS)
    binary = grayscale.point(lambda px: 0 if px < 205 else 255, mode="1")
    return [
        binary,
        binary.rotate(90, expand=True),
        binary.rotate(180, expand=True),
        binary.rotate(270, expand=True),
    ]


def _extract_text_with_confidence(image: Image.Image, ocr_config: OcrConfig) -> OcrExtraction:
    assert pytesseract is not None
    try:
        data = pytesseract.image_to_data(
            image,
            lang=ocr_config.languages,
            config=f"--oem 3 --psm {ocr_config.psm}",
            output_type=TesseractOutput.DICT,
        )
    except Exception:
        return OcrExtraction(text="", confidence=0.0)

    parts: list[str] = []
    confidence_values: list[float] = []
    for text, confidence in zip(data.get("text", []), data.get("conf", [])):
        normalized = normalize_ocr_text(str(text))
        if not normalized:
            continue
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            continue
        if score < 0:
            continue
        parts.append(normalized)
        confidence_values.append(score)

    if not parts:
        return OcrExtraction(text="", confidence=0.0)
    return OcrExtraction(
        text=normalize_ocr_text(" ".join(parts)),
        confidence=sum(confidence_values) / len(confidence_values),
    )


def _ocr_text_score(text: str) -> int:
    return sum(char.isalnum() for char in text)


def _ensure_ocr_ready(ocr_config: OcrConfig) -> None:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed.")
    if ocr_config.mode == "off":
        raise RuntimeError("Template parsing requires OCR. Use ocr mode auto or required.")
    _apply_tesseract_cmd(ocr_config)
    tesseract_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    if ocr_config.tesseract_cmd and not Path(ocr_config.tesseract_cmd).exists():
        raise RuntimeError(f"Tesseract executable not found: {ocr_config.tesseract_cmd}")
    if not ocr_config.tesseract_cmd and not shutil.which(tesseract_cmd) and not shutil.which("tesseract"):
        raise RuntimeError("Tesseract executable is not available in PATH. Set --tesseract-cmd in the GUI.")


def _apply_tesseract_cmd(ocr_config: OcrConfig) -> None:
    if pytesseract is None:
        return
    if ocr_config.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = ocr_config.tesseract_cmd

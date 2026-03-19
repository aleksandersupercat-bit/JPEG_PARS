from __future__ import annotations

import json
import os
import re
import string
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage

from .features import OcrConfig, SUPPORTED_EXTENSIONS, resolve_tesseract_cmd

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
except ImportError:
    pytesseract = None
    TesseractOutput = None


PLUS_MINUS_PATTERNS = [
    re.compile(r"(?<=\d)\s*\+\s*/?\s*-\s*(?=\d)"),
    re.compile(r"(?<=\d)\s*\+\s*-\s*(?=\d)"),
    re.compile(r"(?<=\d)\s*[+\u00b1]\s*(?=\d)"),
]
NOISE_ONLY_PATTERN = re.compile(r"^[\W_]+$")
PADDLE_ENGINE_CACHE: dict[str, object] = {}
PADDLE_ENGINE_LOCK = threading.Lock()
PADDLE_INIT_ERRORS: dict[str, str] = {}
DIMENSION_PATTERNS = [
    re.compile(r"^\d{1,4}\u00b1\d{1,3}$"),
    re.compile(r"^[\u2300\u00d8\u03c6]\d{1,4}\u00b1\d{1,3}$"),
    re.compile(r"^\d{1,4}\u00b1\d{1,3}%$"),
    re.compile(r"^[\u2300\u00d8\u03c6]?\d{1,4}(?:\.\d+)?$"),
]


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
    debug_candidates: dict[str, list["OcrCandidate"]]


@dataclass(slots=True)
class OcrExtraction:
    text: str
    confidence: float


@dataclass(slots=True)
class OcrCandidate:
    variant_name: str
    backend: str
    text: str
    confidence: float
    score: float


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
        debug_candidates: dict[str, list[OcrCandidate]] = {}
        for region in regions:
            extraction, candidates = extract_region_text(image, region, ocr_config)
            values[region.name] = extraction.text
            confidences[region.name] = extraction.confidence
            debug_candidates[region.name] = candidates
        results.append(
            ParsedSheet(
                file_name=path.name,
                file_path=str(path),
                values=values,
                confidences=confidences,
                debug_candidates=debug_candidates,
            )
        )
    return results


def extract_region_text(image: Image.Image, region: TemplateRegion, ocr_config: OcrConfig) -> tuple[OcrExtraction, list["OcrCandidate"]]:
    width, height = image.size
    box = (
        int(region.x0 * width),
        int(region.y0 * height),
        int(region.x1 * width),
        int(region.y1 * height),
    )
    cropped = image.crop(box)
    if cropped.width <= 2 or cropped.height <= 2:
        return OcrExtraction(text="", confidence=0.0), []

    backend = _preferred_backend(ocr_config)
    if backend == "paddle":
        # PaddleOCR has its own text detection: pass the full region crop
        # directly so the model can locate all text lines in one call.
        # Splitting into thin line strips forces extreme rescaling inside
        # PaddleOCR (e.g. a 12 px strip → 1888 px wide after limit_side_len
        # upscaling), which is both slow and inaccurate.
        all_candidates: list[OcrCandidate] = []

        prepared = _prepare_paddle_image(cropped)
        extraction = _extract_with_paddle(prepared, ocr_config)
        if extraction.text:
            all_candidates.append(OcrCandidate(
                variant_name="gray",
                backend="paddle",
                text=extraction.text,
                confidence=extraction.confidence,
                score=score_candidate(extraction.text, extraction.confidence),
            ))

        # For vertically-elongated regions the text is likely written top-to-
        # bottom or bottom-to-top.  Try both 90° rotations and keep whichever
        # produces the highest score.
        region_w = region.x1 - region.x0
        region_h = region.y1 - region.y0
        if region_h > region_w * 1.5:
            for degrees, vname in [(90, "vertical_ccw"), (-90, "vertical_cw")]:
                rotated = _prepare_paddle_image(cropped.rotate(degrees, expand=True))
                ext_rot = _extract_with_paddle(rotated, ocr_config)
                if ext_rot.text:
                    all_candidates.append(OcrCandidate(
                        variant_name=vname,
                        backend="paddle",
                        text=ext_rot.text,
                        confidence=ext_rot.confidence,
                        score=score_candidate(ext_rot.text, ext_rot.confidence),
                    ))

        if not all_candidates:
            return OcrExtraction(text="", confidence=0.0), []
        best = max(all_candidates, key=lambda c: c.score)
        return (
            OcrExtraction(text=best.text, confidence=best.confidence),
            sorted(all_candidates, key=lambda c: c.score, reverse=True),
        )

    # Tesseract path: split into individual text lines first.
    inner = _crop_inner_border(cropped)
    lines = _split_text_lines(inner)
    if not lines:
        lines = [inner]

    line_results: list[OcrExtraction] = []
    all_candidates: list[OcrCandidate] = []
    for line in lines:
        extraction, candidates = _recognize_line(line, ocr_config)
        if extraction.text:
            line_results.append(extraction)
        all_candidates.extend(candidates)

    if not line_results:
        return OcrExtraction(text="", confidence=0.0), all_candidates
    merged_text = normalize_ocr_text(" ".join(item.text for item in line_results))
    merged_conf = sum(item.confidence for item in line_results) / len(line_results)
    return OcrExtraction(text=merged_text, confidence=merged_conf), sorted(
        all_candidates,
        key=lambda item: item.score,
        reverse=True,
    )


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
    compact = " ".join(text.replace("\n", " ").replace("\r", " ").split()).strip()
    return _normalize_measurement_text(compact)


def score_candidate(text: str, confidence: float) -> float:
    score = confidence
    if any(pattern.match(text) for pattern in DIMENSION_PATTERNS):
        score += 40.0
    score += sum(char.isdigit() for char in text) * 1.2
    if "\u00b1" in text:
        score += 15.0
    if "%" in text:
        score += 8.0
    bad_chars = sum(1 for char in text if char not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя\u00b1\u2300\u00d8\u03c6%.-")
    score -= bad_chars * 6.0
    return score


def _recognize_line(image: Image.Image, ocr_config: OcrConfig) -> tuple[OcrExtraction, list[OcrCandidate]]:
    candidates: list[OcrCandidate] = []
    backend = _preferred_backend(ocr_config)
    if backend == "paddle":
        # PaddleOCR works best on its native-resolution grayscale input.
        # Upscaling (4-6×) balloons the image to >1400 px wide, which makes
        # the detection model run 6× slower and gives no accuracy benefit.
        # One call with minimal preprocessing is enough.
        prepared = _prepare_paddle_image(image)
        extraction = _extract_with_paddle(prepared, ocr_config)
        if extraction.text:
            candidates.append(
                OcrCandidate(
                    variant_name="gray",
                    backend="paddle",
                    text=extraction.text,
                    confidence=extraction.confidence,
                    score=score_candidate(extraction.text, extraction.confidence),
                )
            )
    elif backend == "tesseract":
        for variant_name, variant in _preprocess_variants(image):
            extraction = _extract_with_tesseract(variant, ocr_config)
            if extraction.text:
                candidates.append(
                    OcrCandidate(
                        variant_name=variant_name,
                        backend="tesseract",
                        text=extraction.text,
                        confidence=extraction.confidence,
                        score=score_candidate(extraction.text, extraction.confidence),
                    )
                )
    if not candidates:
        return OcrExtraction(text="", confidence=0.0), []
    best = max(candidates, key=lambda item: item.score)
    return OcrExtraction(text=best.text, confidence=best.confidence), sorted(candidates, key=lambda item: item.score, reverse=True)


def _crop_inner_border(image: Image.Image) -> Image.Image:
    width, height = image.size
    pad_x = max(2, int(width * 0.03))
    pad_y = max(2, int(height * 0.08))
    if width - pad_x * 2 <= 2 or height - pad_y * 2 <= 2:
        return image
    return image.crop((pad_x, pad_y, width - pad_x, height - pad_y))


def _preprocess_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
    base = _prepare_base_image(image)
    variants = [("gray_up", base)]

    sharp = ImageEnhance.Sharpness(base).enhance(2.0)
    variants.append(("gray_up_sharp", sharp))

    gray_data = np.asarray(base, dtype=np.uint8)
    otsu = _otsu_threshold(gray_data)
    variants.append(("otsu", Image.fromarray(otsu, mode="L")))

    adaptive = _adaptive_threshold(gray_data, block_size=31, offset=11)
    variants.append(("adaptive", Image.fromarray(adaptive, mode="L")))

    variants.append(("otsu_inv", Image.fromarray(255 - otsu, mode="L")))

    close = ndimage.binary_closing(otsu < 128, structure=np.ones((2, 2), dtype=bool))
    variants.append(("close", Image.fromarray(np.where(close, 0, 255).astype(np.uint8), mode="L")))
    return variants


def _prepare_base_image(image: Image.Image) -> Image.Image:
    inner = _crop_inner_border(image)
    gray = ImageOps.autocontrast(inner.convert("L"))
    upscale = 6 if max(gray.size) < 220 else 4
    gray = gray.resize((max(1, gray.width * upscale), max(1, gray.height * upscale)), Image.Resampling.LANCZOS)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = ImageEnhance.Sharpness(gray).enhance(2.4)
    if gray.height > 100:
        gray = _remove_long_lines(gray)
    return gray


def _prepare_ocr_image(image: Image.Image) -> Image.Image:
    return _prepare_base_image(image)


def _prepare_paddle_image(image: Image.Image) -> Image.Image:
    """Minimal preprocessing for PaddleOCR.

    PaddleOCR runs its own internal resize and normalisation before
    detection/recognition, so there is no need to upscale 4-6× here.
    Upscaling only makes the server-side detection model process a much
    larger image (e.g. 1416×48 instead of 354×12), which is ~6× slower
    with no accuracy gain.  We only strip the inner border and apply
    autocontrast so the model gets clean, evenly-lit input.
    """
    inner = _crop_inner_border(image)
    gray = ImageOps.autocontrast(inner.convert("L"))
    # PaddleOCR's text detection needs a minimum side of ~32 px.
    min_side = min(gray.size)
    if min_side < 32:
        scale = max(2, 32 // min_side + 1)
        gray = gray.resize(
            (max(1, gray.width * scale), max(1, gray.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return gray


def _remove_long_lines(image: Image.Image) -> Image.Image:
    data = np.asarray(image, dtype=np.uint8)
    binary = data < min(220, int(np.percentile(data, 78)))
    horizontal_len = max(18, binary.shape[1] // 5)
    vertical_len = max(18, binary.shape[0] // 5)
    horizontal_lines = ndimage.binary_opening(binary, structure=np.ones((1, horizontal_len), dtype=bool))
    vertical_lines = ndimage.binary_opening(binary, structure=np.ones((vertical_len, 1), dtype=bool))
    cleaned = binary & ~(horizontal_lines | vertical_lines)

    labeled, count = ndimage.label(cleaned)
    if count > 0:
        objects = ndimage.find_objects(labeled)
        filtered = np.zeros_like(cleaned)
        for index, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            component = labeled[slices] == index
            area = int(component.sum())
            if area < 8:
                continue
            height, width = component.shape
            ratio = max(width / max(height, 1), height / max(width, 1))
            if ratio > 8 and area < 40:
                continue
            filtered[slices] |= component
        cleaned = filtered

    if cleaned.sum() < 8:
        cleaned = binary
    cleaned = ndimage.binary_closing(cleaned, structure=np.ones((2, 2), dtype=bool))
    return Image.fromarray(np.where(cleaned, 0, 255).astype(np.uint8), mode="L")


def _split_text_lines(image: Image.Image) -> list[Image.Image]:
    data = np.asarray(image.convert("L"), dtype=np.uint8)
    binary = data < 180
    row_projection = binary.sum(axis=1)
    threshold = max(1, int(row_projection.max() * 0.12))
    active = row_projection >= threshold

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= 6:
                segments.append((start, index))
            start = None
    if start is not None and len(active) - start >= 6:
        segments.append((start, len(active)))

    if not segments:
        return []

    results: list[Image.Image] = []
    for top, bottom in segments[:3]:
        pad = 4
        cropped = image.crop((0, max(0, top - pad), image.width, min(image.height, bottom + pad)))
        if cropped.height > 4:
            results.append(cropped)
    return results


def _extract_with_paddle(image: Image.Image, ocr_config: OcrConfig) -> OcrExtraction:
    engine = _get_paddle_engine(ocr_config)
    if engine is None:
        return OcrExtraction(text="", confidence=0.0)
    paddle_input = np.asarray(image.convert("RGB"))
    try:
        result = engine.predict(
            paddle_input,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=0.0,
        )
    except Exception:
        return OcrExtraction(text="", confidence=0.0)
    # PP-OCRv3 lacks ± in its vocabulary and silently drops the glyph, merging
    # adjacent digits (e.g. "86±10%kg" → "8610%kg").  Detect the ± visually
    # and pass its x-position so _parse_paddle_result can inject it at the
    # correct character index using the recognised text-box coordinates.
    pm_x = _plus_minus_glyph_x(image)
    extraction = _parse_paddle_result(result, pm_x=pm_x)
    return _postprocess_extraction(extraction, image)


def _extract_with_tesseract(image: Image.Image, ocr_config: OcrConfig) -> OcrExtraction:
    _apply_tesseract_cmd(ocr_config)
    if pytesseract is None:
        return OcrExtraction(text="", confidence=0.0)
    try:
        data = pytesseract.image_to_data(
            image,
            lang=ocr_config.languages,
            config=_build_tesseract_config(7),
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
    extraction = OcrExtraction(
        text=normalize_ocr_text(" ".join(parts)),
        confidence=sum(confidence_values) / len(confidence_values),
    )
    return _postprocess_extraction(extraction, image)


def _postprocess_extraction(extraction: OcrExtraction, image: Image.Image) -> OcrExtraction:
    if _contains_plus_minus_symbol(image):
        # Pass the raw text to _merge_plus_minus_tokens BEFORE normalize_ocr_text
        # removes spaces. If Paddle skips ± and outputs "120 0.5", the space is
        # the only separator available — normalize_ocr_text would collapse it to
        # "1200.5" first, leaving nothing to merge on.
        text = _merge_plus_minus_tokens(extraction.text)
    else:
        text = normalize_ocr_text(extraction.text)
    if _looks_like_noise(text):
        return OcrExtraction(text="", confidence=0.0)
    return OcrExtraction(text=text, confidence=extraction.confidence)


def _parse_paddle_result(result: object, pm_x: int | None = None) -> OcrExtraction:
    texts: list[str] = []
    confidences: list[float] = []
    if not isinstance(result, list):
        return OcrExtraction(text="", confidence=0.0)

    def walk(node: object) -> None:
        if isinstance(node, dict):
            rec_texts = node.get("rec_texts")
            rec_scores = node.get("rec_scores")
            rec_boxes = node.get("rec_boxes")
            if isinstance(rec_texts, list):
                for index, text in enumerate(rec_texts):
                    if isinstance(text, str):
                        # When PP-OCRv3 drops the ± glyph and merges adjacent
                        # digits (e.g. "8610%kg"), inject ± at the character
                        # position that matches the visually-detected glyph's
                        # x-coordinate within the recognised text box.
                        # rec_boxes may be a numpy array or a plain list.
                        if pm_x is not None and rec_boxes is not None and index < len(rec_boxes):
                            try:
                                box = rec_boxes[index]
                                bx0 = int(box[0]); bx1 = int(box[2])
                                if bx0 <= pm_x <= bx1 and len(text) > 0:
                                    rel = (pm_x - bx0) / max(1, bx1 - bx0)
                                    idx = max(1, min(len(text), int(rel * len(text) + 0.5)))
                                    text = text[:idx] + "\u00b1" + text[idx:]
                            except (IndexError, TypeError, ValueError):
                                pass
                        texts.append(text)
                        score = 0.0
                        if isinstance(rec_scores, list) and index < len(rec_scores):
                            try:
                                score = float(rec_scores[index])
                            except (TypeError, ValueError):
                                score = 0.0
                        confidences.append(score * 100.0 if score <= 1.0 else score)
            for value in node.values():
                walk(value)
            return
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], str):
            texts.append(node[0])
            try:
                score = float(node[1])
            except (TypeError, ValueError):
                score = 0.0
            confidences.append(score * 100.0 if score <= 1.0 else score)
            return
        if isinstance(node, list):
            for child in node:
                walk(child)

    walk(result)
    if not texts:
        return OcrExtraction(text="", confidence=0.0)
    return OcrExtraction(
        text=normalize_ocr_text(" ".join(texts)),
        confidence=sum(confidences) / len(confidences),
    )


def _get_paddle_engine(ocr_config: OcrConfig) -> object | None:
    if PaddleOCR is None:
        return None
    lang = _resolve_paddle_lang(ocr_config.languages)
    version = getattr(ocr_config, "ocr_version", "PP-OCRv3") or "PP-OCRv3"
    cache_key = f"{lang}:{version}"
    with PADDLE_ENGINE_LOCK:
        if cache_key in PADDLE_ENGINE_CACHE:
            return PADDLE_ENGINE_CACHE[cache_key]
        kwargs: dict = dict(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        if version:
            kwargs["ocr_version"] = version
        try:
            engine = PaddleOCR(**kwargs)
        except Exception as exc:
            PADDLE_INIT_ERRORS[cache_key] = f"{type(exc).__name__}: {exc}"
            return None
        PADDLE_INIT_ERRORS.pop(cache_key, None)
        PADDLE_ENGINE_CACHE[cache_key] = engine
        return engine


def _resolve_paddle_lang(languages: str) -> str:
    tokens = {token.strip().lower() for token in languages.split("+") if token.strip()}
    # For dimension zones, English digits/symbol models are usually cleaner than mixed Cyrillic.
    if tokens <= {"eng", "en", "rus", "ru"}:
        return "en"
    if {"rus", "ru"} & tokens and not ({"eng", "en"} & tokens):
        return "ru"
    return "en"


def _preferred_backend(ocr_config: OcrConfig) -> str:
    if _get_paddle_engine(ocr_config) is not None:
        return "paddle"
    if pytesseract is not None and resolve_tesseract_cmd(ocr_config.tesseract_cmd):
        return "tesseract"
    return "none"


def _normalize_measurement_text(text: str) -> str:
    normalized = text.strip()
    replacements = {
        "=": "\u00b1",
        "+-": "\u00b1",
        "土": "\u00b1",
        "\u2213": "\u00b1",
        "\u2014": "-",
        "\u2013": "-",
        "Ф": "\u2300",
        "φ": "\u2300",
        "Ø": "\u2300",
        # PP-OCRv3 misreads the diameter symbol ∅ as $ (circle + stroke confusion)
        "$": "\u2300",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.replace("°/o", "%")
    for pattern in PLUS_MINUS_PATTERNS:
        normalized = pattern.sub("\u00b1", normalized)
    normalized = re.sub(r"\s*%\s*", "%", normalized)
    normalized = re.sub(r"\s*\u00b1\s*", "\u00b1", normalized)
    normalized = normalized.replace(" ", "")
    return normalized


def _merge_plus_minus_tokens(text: str) -> str:
    merged = re.sub(r"(?<=\d)\s*[+=~-]\s*(?=\d)", "\u00b1", text)
    merged = re.sub(r"(?<=\d)\s+(?=\d{1,3}(?:\D|$))", "\u00b1", merged, count=1)
    return normalize_ocr_text(merged)


def _plus_minus_glyph_x(image: Image.Image) -> int | None:
    """Return the x-centre pixel of the ± glyph in *image*, or None if absent.

    Strategy
    --------
    1. Binarise the grayscale image.
    2. Apply a narrow vertical dilation (5×1) so disconnected strokes of the
       same glyph (e.g. the three horizontal bars of ±) merge into one labeled
       region, while adjacent characters (separated horizontally) are unaffected.
    3. For each labeled region that has a roughly square aspect ratio, analyse
       the *original* (undilated) binary within its bounding box.  The ± glyph
       produces ≥2 horizontal "runs" of ink (top bar + crossbar, or all three
       bars) with a dominant central vertical bar, which distinguishes it from
       letters and digits.
    """
    data = np.asarray(image.convert("L"), dtype=np.uint8)
    binary = data < 128
    if not np.any(binary):
        return None
    merged = ndimage.binary_dilation(binary, structure=np.ones((5, 1), dtype=bool))
    labeled, count = ndimage.label(merged)
    if count == 0:
        return None
    for component_idx in range(1, count + 1):
        ys, xs = np.nonzero(labeled == component_idx)
        if len(ys) < 12:
            continue
        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1
        height = bottom - top
        width = right - left
        if height < 6 or width < 4:
            continue
        ratio = width / max(height, 1)
        if ratio < 0.15 or ratio > 2.2:
            continue
        mask = binary[top:bottom, left:right]
        hproj = mask.sum(axis=1).astype(float)
        vproj = mask.sum(axis=0).astype(float)
        if hproj.max() == 0:
            continue
        h_threshold = max(1, int(hproj.max() * 0.42))
        v_threshold = max(1, int(vproj.max() * 0.42))
        h_runs = _count_runs(hproj >= h_threshold)
        v_runs = _count_runs(vproj >= v_threshold)
        dominant_column = int(np.argmax(vproj))
        centered = width * 0.2 <= dominant_column <= width * 0.8
        if not centered:
            continue
        # Three horizontal bars → unambiguous ±
        if h_runs >= 3 and v_runs >= 1:
            return (left + right) // 2
        # Two bars ("+" part of ±, minus bar may be a separate component):
        # require a strong central vertical-bar peak.  For ± the center column
        # vproj is typically 2-4× the column average because the thin vertical
        # bar runs through the full glyph height; round letters like "g"/"d"
        # have a much flatter column distribution and are rejected here.
        if h_runs == 2 and vproj.mean() > 0:
            peak_ratio = float(vproj[dominant_column]) / float(vproj.mean())
            if peak_ratio >= 1.8:
                return (left + right) // 2
    return None


def _contains_plus_minus_symbol(image: Image.Image) -> bool:
    return _plus_minus_glyph_x(image) is not None


def _looks_like_noise(text: str) -> bool:
    if not text:
        return True
    if NOISE_ONLY_PATTERN.match(text):
        return True
    if len(text) <= 2 and not any(char.isalnum() for char in text):
        return True
    return False


def _count_runs(values: np.ndarray) -> int:
    runs = 0
    in_run = False
    for value in values.tolist():
        if value and not in_run:
            runs += 1
            in_run = True
        elif not value:
            in_run = False
    return runs


def _unique_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _psm_candidates(base_psm: int) -> list[int]:
    return _unique_preserve_order([base_psm, 5, 6, 7, 11, 12, 13])


def _build_tesseract_config(psm: int) -> str:
    return (
        f"--oem 3 --psm {psm} "
        "-c preserve_interword_spaces=1 "
        "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        ".,:;+-\u00b1*/()[]{}<>=%\u2300\u00d8\u03c6"
    )


def _otsu_threshold(gray: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_background = 0.0
    weight_background = 0.0
    max_variance = -1.0
    threshold = 0
    for value in range(256):
        weight_background += hist[value]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += value * hist[value]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = value
    return np.where(gray > threshold, 255, 0).astype(np.uint8)


def _adaptive_threshold(gray: np.ndarray, block_size: int = 31, offset: int = 11) -> np.ndarray:
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    local_mean = ndimage.uniform_filter(gray.astype(np.float32), size=block_size)
    return np.where(gray > (local_mean - offset), 255, 0).astype(np.uint8)


def _ensure_ocr_ready(ocr_config: OcrConfig) -> None:
    if _get_paddle_engine(ocr_config) is not None:
        return
    cache_key = f"{_resolve_paddle_lang(ocr_config.languages)}:{getattr(ocr_config, 'ocr_version', 'PP-OCRv3')}"
    if pytesseract is None:
        details = PADDLE_INIT_ERRORS.get(cache_key)
        if details:
            raise RuntimeError(f"PaddleOCR failed to initialize: {details}")
        raise RuntimeError("No OCR backend is installed. Install PaddleOCR or Tesseract.")
    resolved = resolve_tesseract_cmd(ocr_config.tesseract_cmd)
    if not resolved:
        details = PADDLE_INIT_ERRORS.get(cache_key)
        if details:
            raise RuntimeError(f"PaddleOCR failed to initialize: {details}")
        raise RuntimeError("No OCR backend is available. Install PaddleOCR or set tesseract.exe.")
    pytesseract.pytesseract.tesseract_cmd = resolved


def get_template_ocr_backend_info(ocr_config: OcrConfig) -> tuple[str, str]:
    if _get_paddle_engine(ocr_config) is not None:
        version = getattr(ocr_config, "ocr_version", "PP-OCRv3") or "PP-OCRv3"
        return "paddle", f"PaddleOCR {version}"
    cache_key = f"{_resolve_paddle_lang(ocr_config.languages)}:{getattr(ocr_config, 'ocr_version', 'PP-OCRv3')}"
    if pytesseract is not None and resolve_tesseract_cmd(ocr_config.tesseract_cmd):
        details = PADDLE_INIT_ERRORS.get(cache_key)
        if details:
            return "tesseract", f"Tesseract fallback, Paddle init error: {details}"
        return "tesseract", "Tesseract fallback"
    details = PADDLE_INIT_ERRORS.get(cache_key)
    if details:
        return "none", f"Paddle init error: {details}"
    return "none", "No OCR backend available"


def _apply_tesseract_cmd(ocr_config: OcrConfig) -> None:
    if pytesseract is None:
        return
    resolved = resolve_tesseract_cmd(ocr_config.tesseract_cmd)
    if resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved

from __future__ import annotations

import json
import logging
import os
import re
import string
import threading
import time
import unicodedata
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

_LOG = logging.getLogger("jpeg_pars.ocr")

# Special characters we care about — shown with name in debug output
_SPECIAL_CHARS: dict[str, str] = {
    "\u00b1": "±",
    "\u2300": "∅",
    "\u00d8": "Ø",
    "\u03c6": "φ",
    "%": "%",
}


def _repr_char(c: str) -> str:
    """Return human-readable representation of a single character."""
    if c in _SPECIAL_CHARS:
        return f"[{_SPECIAL_CHARS[c]} U+{ord(c):04X}]"
    if ord(c) > 127:
        try:
            name = unicodedata.name(c, "?")
        except Exception:
            name = "?"
        return f"[U+{ord(c):04X} {name}]"
    return c


def _repr_text(text: str) -> str:
    """Annotate non-ASCII / special chars in a string for log output."""
    parts: list[str] = []
    for c in text:
        if c in _SPECIAL_CHARS or ord(c) > 127:
            parts.append(_repr_char(c))
        else:
            parts.append(c)
    return "".join(parts)


def _score_breakdown(text: str, confidence: float) -> str:
    """Return a human-readable score breakdown string."""
    dim_match = any(p.match(text) for p in DIMENSION_PATTERNS)
    pm_bonus = 15.0 if "\u00b1" in text else 0.0
    pct_bonus = 8.0 if "%" in text else 0.0
    digit_bonus = sum(c.isdigit() for c in text) * 1.2
    bad_chars = sum(
        1 for c in text
        if c not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "\u00b1\u2300\u00d8\u03c6%.-"
    )
    bad_penalty = bad_chars * 6.0
    total = confidence + (40.0 if dim_match else 0.0) + pm_bonus + pct_bonus + digit_bonus - bad_penalty
    parts = [f"conf={confidence:.1f}"]
    if dim_match:
        parts.append("+40(dim_match)")
    if pm_bonus:
        parts.append(f"+{pm_bonus:.0f}(±)")
    if pct_bonus:
        parts.append(f"+{pct_bonus:.0f}(%)")
    if digit_bonus:
        parts.append(f"+{digit_bonus:.1f}(digits)")
    if bad_penalty:
        parts.append(f"-{bad_penalty:.0f}(bad×{bad_chars})")
    parts.append(f"= {total:.1f}")
    return " ".join(parts)


# Characters that OCR commonly confuses with ± (U+00B1)
_PM_LOOKALIKES: dict[str, str] = {
    "+": "plus without minus bar",
    "7": "digit 7 (bar on stem)",
    "1": "digit 1",
    "I": "letter I",
    "l": "letter l",
    "t": "letter t",
    "T": "letter T",
    "=": "equals (two bars)",
}


def _log_char_analysis(region_name: str, raw_words: list["OcrCandidate"], final_text: str) -> None:
    """Log per-character breakdown — focus on where ± might have been lost."""
    if not _LOG.isEnabledFor(logging.DEBUG):
        return
    _LOG.debug("   [%s] ── char analysis ──────────────────────────────────", region_name)
    # Show final result character by character
    _LOG.debug("   [%s] final text: %r  (%d chars)",
               region_name, _repr_text(final_text), len(final_text))
    for i, c in enumerate(final_text):
        status = "OK" if c not in _PM_LOOKALIKES else f"SUSPECT — {_PM_LOOKALIKES[c]}"
        _LOG.debug("   [%s]   char[%d] = %-4s  U+%04X  %-20s  %s",
                   region_name, i, repr(c), ord(c),
                   unicodedata.name(c, "?")[:20], status)
    has_pm = "\u00b1" in final_text
    _LOG.debug("   [%s] ± present in final: %s", region_name, "YES ✓" if has_pm else "NO ✗")
    if not has_pm:
        # Look in raw Tesseract words for what might have replaced it
        _LOG.debug("   [%s] searching raw words for ± lookalikes:", region_name)
        for w in raw_words:
            if w.variant_name == "tess_raw":
                for c in w.text:
                    if c in _PM_LOOKALIKES:
                        _LOG.debug("   [%s]   tess_raw word=%r  char=%r → might be ±  (%s)",
                                   region_name, w.text, c, _PM_LOOKALIKES[c])
    _LOG.debug("   [%s] ────────────────────────────────────────────────────", region_name)


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
    re.compile(r"^\d{1,4}\u00b1\d{1,3}%[a-zA-Z]{1,4}$"),
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
    _LOG.info("parse_template_batch: %d file(s), %d region(s), backend=%s, model=%s",
              len(files), len(regions), _preferred_backend(ocr_config), ocr_config.ocr_version)
    results: list[ParsedSheet] = []
    for path in files:
        t_file = time.perf_counter()
        image = Image.open(path).convert("RGB")
        _LOG.info("── FILE: %s  size=%dx%d", path.name, image.width, image.height)
        values: dict[str, str] = {}
        confidences: dict[str, float] = {}
        debug_candidates: dict[str, list[OcrCandidate]] = {}
        for region in regions:
            t_region = time.perf_counter()
            extraction, candidates = extract_region_text(
                image, region, ocr_config, full_page_cache=None
            )
            elapsed = (time.perf_counter() - t_region) * 1000
            result_repr = _repr_text(extraction.text) if extraction.text else "(empty)"
            _LOG.info("   region %-4s → %-30s  conf=%.1f  %.0fms",
                      region.name, result_repr, extraction.confidence, elapsed)
            values[region.name] = extraction.text
            confidences[region.name] = extraction.confidence
            debug_candidates[region.name] = candidates
        _LOG.info("── FILE done: %.0fms", (time.perf_counter() - t_file) * 1000)
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


def extract_region_text(
    image: Image.Image,
    region: TemplateRegion,
    ocr_config: OcrConfig,
    full_page_cache: object = None,
) -> tuple[OcrExtraction, list["OcrCandidate"]]:
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
    _LOG.debug("   [%s] box=%s  crop=%dx%d  backend=%s",
               region.name, box, cropped.width, cropped.height, backend)
    if backend == "paddle":
        all_candidates: list[OcrCandidate] = []

        t0 = time.perf_counter()
        prepared = _prepare_paddle_image(cropped)
        extraction = _extract_with_paddle(prepared, ocr_config)
        _LOG.debug("   [%s] paddle_crop: %.0fms  crop=%dx%d  raw=%r  conf=%.1f",
                   region.name, (time.perf_counter() - t0) * 1000,
                   cropped.width, cropped.height,
                   extraction.text, extraction.confidence)

        if extraction.text:
            sc = score_candidate(extraction.text, extraction.confidence)
            _LOG.debug("   [%s] paddle candidate: %s  score=(%s)",
                       region.name, _repr_text(extraction.text),
                       _score_breakdown(extraction.text, extraction.confidence))
            all_candidates.append(OcrCandidate(
                variant_name="paddle",
                backend="paddle",
                text=extraction.text,
                confidence=extraction.confidence,
                score=sc,
            ))

        # For vertically-elongated regions try both 90° rotations.
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

        # Tesseract supplementary pass: run one variant so its output is
        # visible in OCR Debug and can rescue ± that Paddle dropped.
        t0 = time.perf_counter()
        tess_extraction, tess_raw = _tesseract_quick_pass(cropped, ocr_config)
        _LOG.debug("   [%s] tess_quick: %.0fms  result=%r  conf=%.1f  words=%d",
                   region.name, (time.perf_counter() - t0) * 1000,
                   tess_extraction.text, tess_extraction.confidence, len(tess_raw))
        if tess_raw:
            _LOG.debug("   [%s] tess_words:", region.name)
            for w in tess_raw:
                pm_flag = " ← [± MISSED?]" if w.text.strip() in ("7", "1", "I", "l", "+") else ""
                _LOG.debug("         word %-12s conf=%5.1f  repr=%s%s",
                           repr(w.text), w.confidence, _repr_text(w.text), pm_flag)
            all_candidates.extend(tess_raw)
        if tess_extraction.text:
            _LOG.debug("   [%s] tess candidate: %s  score=(%s)",
                       region.name, _repr_text(tess_extraction.text),
                       _score_breakdown(tess_extraction.text, tess_extraction.confidence))
            all_candidates.append(OcrCandidate(
                variant_name="tess_gray",
                backend="tesseract",
                text=tess_extraction.text,
                confidence=tess_extraction.confidence,
                score=score_candidate(tess_extraction.text, tess_extraction.confidence),
            ))

        if not all_candidates:
            _LOG.debug("   [%s] no candidates → empty", region.name)
            return OcrExtraction(text="", confidence=0.0), []
        real_candidates = [c for c in all_candidates if c.variant_name != "tess_raw"]
        best = max(real_candidates, key=lambda c: c.score) if real_candidates else max(all_candidates, key=lambda c: c.score)
        _LOG.debug("   [%s] WINNER: backend=%-10s variant=%-14s text=%s  score=(%.1f)",
                   region.name, best.backend, best.variant_name,
                   _repr_text(best.text), best.score)
        raw_words = [c for c in all_candidates if c.variant_name == "tess_raw"]
        _log_char_analysis(region.name, raw_words, best.text)
        return (
            OcrExtraction(text=best.text, confidence=best.confidence),
            sorted(all_candidates, key=lambda c: c.score, reverse=True),
        )

    # Tesseract path: process the whole region — splitting into narrow
    # horizontal strips caused Tesseract to read garbage because the strips
    # were too thin and contained confusing vertical border artefacts.
    inner = _crop_inner_border(cropped)
    _LOG.debug("   [%s] tesseract: inner=%dx%d", region.name, inner.width, inner.height)
    extraction, all_candidates = _recognize_line(inner, ocr_config)
    _LOG.debug("   [%s] tesseract result: %r  conf=%.1f", region.name, extraction.text, extraction.confidence)
    raw_words = [c for c in all_candidates if c.variant_name == "tess_raw"]
    _log_char_analysis(region.name, raw_words, extraction.text)
    return extraction, sorted(all_candidates, key=lambda c: c.score, reverse=True)


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
        raw_words_added = False
        for variant_name, variant in _preprocess_variants(image):
            extraction, raw_cands = _extract_with_tesseract(variant, ocr_config)
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
            # Add raw word-level output from the first variant to debug window
            if not raw_words_added and raw_cands:
                candidates.extend(raw_cands)
                raw_words_added = True
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
    orig_w, orig_h = inner.size
    gray = ImageOps.autocontrast(inner.convert("L"))
    upscale = 6 if max(gray.size) < 220 else 4
    gray = gray.resize((max(1, gray.width * upscale), max(1, gray.height * upscale)), Image.Resampling.LANCZOS)
    # Note: MedianFilter and heavy sharpening were removed — they degraded OCR
    # quality on technical fonts (digits + ± glyph).  Autocontrast + upscale is
    # sufficient.  _remove_long_lines is kept only for wide images that are
    # likely to contain actual table borders (orig_w > 300px).
    if gray.height > 100 and orig_w > 300:
        _LOG.debug("      _remove_long_lines: orig_w=%d (applied)", orig_w)
        gray = _remove_long_lines(gray)
    else:
        _LOG.debug("      _remove_long_lines: orig_w=%d (skipped)", orig_w)
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
    # Use 85% of dimension: character strokes span ~50-75% of image height/width,
    # actual cell borders span 100%.  The old width//5 (20%) threshold was
    # removing character strokes and the ± horizontal bar.
    horizontal_len = max(40, binary.shape[1] * 17 // 20)  # 85 % of width
    vertical_len   = max(40, binary.shape[0] * 17 // 20)  # 85 % of height
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


_PADDLE_MAX_SIDE = 4000


def _precompute_full_page_paddle(image: Image.Image, ocr_config: OcrConfig) -> object:
    """Run PaddleOCR once on the full image and return the raw result for reuse.

    Pre-scales the image to fit within _PADDLE_MAX_SIDE so PaddleOCR does not
    need to resize internally (suppresses the "exceeds max_side_limit" message).
    Returns None if Paddle is not available so callers fall back to per-crop mode.
    Returns a dict {"result": ..., "scale": float} on success.
    """
    engine = _get_paddle_engine(ocr_config)
    if engine is None:
        return None
    img = image.convert("RGB")
    w, h = img.size
    scale = min(_PADDLE_MAX_SIDE / w, _PADDLE_MAX_SIDE / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    paddle_input = np.asarray(img)
    try:
        result = engine.predict(
            paddle_input,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=0.0,
        )
        return {"result": result, "scale": scale}
    except Exception:
        return None


def _extract_region_from_cache(
    cached_result: object,
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> OcrExtraction:
    """Extract text for one region from a cached full-page PaddleOCR result.

    Walks the result tree, collects text boxes whose centre falls inside *box*,
    and returns a merged OcrExtraction.
    """
    if not isinstance(cached_result, dict):
        return OcrExtraction(text="", confidence=0.0)
    raw_result = cached_result["result"]
    scale: float = cached_result.get("scale", 1.0)
    # Scale region box to the coordinate space PaddleOCR used.
    rx0, ry0, rx1, ry1 = (c * scale for c in box)
    texts: list[str] = []
    confidences: list[float] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            rec_texts = node.get("rec_texts")
            rec_scores = node.get("rec_scores")
            rec_boxes = node.get("rec_boxes")
            if isinstance(rec_texts, list) and rec_boxes is not None:
                for idx, text in enumerate(rec_texts):
                    if not isinstance(text, str):
                        continue
                    try:
                        raw_box = rec_boxes[idx]
                        # True centre of flat [x0,y0,x1,y1] OR 4-point polygon.
                        coords = np.asarray(raw_box, dtype=float).reshape(-1)
                        n = coords.size
                        if n >= 8:
                            # 4-point polygon [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
                            cx = coords[0::2].mean()
                            cy = coords[1::2].mean()
                        elif n >= 4:
                            cx = (coords[0] + coords[2]) / 2.0
                            cy = (coords[1] + coords[3]) / 2.0
                        else:
                            continue
                    except Exception:
                        continue
                    inside = rx0 <= cx <= rx1 and ry0 <= cy <= ry1
                    _LOG.debug("      cache_box: %r  cx=%.0f cy=%.0f  region=(%.0f,%.0f,%.0f,%.0f)  %s",
                               text, cx, cy, rx0, ry0, rx1, ry1,
                               "HIT" if inside else "miss")
                    if inside:
                        texts.append(text)
                        score = 0.0
                        if isinstance(rec_scores, list) and idx < len(rec_scores):
                            try:
                                score = float(rec_scores[idx])
                            except (TypeError, ValueError):
                                pass
                        confidences.append(score * 100.0 if score <= 1.0 else score)
            for v in node.values():
                walk(v)
            return
        if isinstance(node, list):
            for child in node:
                walk(child)

    walk(raw_result)
    if not texts:
        return OcrExtraction(text="", confidence=0.0)
    return _postprocess_extraction(OcrExtraction(
        text=normalize_ocr_text(" ".join(texts)),
        confidence=sum(confidences) / len(confidences),
    ))


def _tesseract_quick_pass(
    image: Image.Image, ocr_config: OcrConfig
) -> tuple[OcrExtraction, list["OcrCandidate"]]:
    """Run one Tesseract variant on *image* for debug visibility in Paddle mode.

    Uses the upscaled-gray variant only (no multi-variant sweep) to keep
    overhead low.  Returns (extraction, raw_word_candidates).
    """
    if pytesseract is None or not resolve_tesseract_cmd(ocr_config.tesseract_cmd):
        return OcrExtraction(text="", confidence=0.0), []
    inner = _crop_inner_border(image)
    gray = ImageOps.autocontrast(inner.convert("L"))
    upscale = 6 if max(gray.size) < 220 else 4
    gray = gray.resize(
        (max(1, gray.width * upscale), max(1, gray.height * upscale)),
        Image.Resampling.LANCZOS,
    )
    return _extract_with_tesseract(gray, ocr_config)


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
    extraction = _parse_paddle_result(result)
    return _postprocess_extraction(extraction)


def _extract_with_tesseract(
    image: Image.Image, ocr_config: OcrConfig
) -> tuple[OcrExtraction, list["OcrCandidate"]]:
    _apply_tesseract_cmd(ocr_config)
    if pytesseract is None:
        return OcrExtraction(text="", confidence=0.0), []
    try:
        data = pytesseract.image_to_data(
            image,
            lang=ocr_config.languages,
            config=_build_tesseract_config(7),
            output_type=TesseractOutput.DICT,
        )
    except Exception:
        return OcrExtraction(text="", confidence=0.0), []

    parts: list[str] = []
    confidence_values: list[float] = []
    raw_candidates: list[OcrCandidate] = []
    for text, confidence in zip(data.get("text", []), data.get("conf", [])):
        raw = str(text).strip()
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = -1.0
        normalized = normalize_ocr_text(raw)
        if raw:
            raw_candidates.append(OcrCandidate(
                variant_name="tess_raw",
                backend="tesseract",
                text=normalized if normalized else raw,  # show normalized so ± is visible
                confidence=max(0.0, score),
                score=max(0.0, score),
            ))
        if not normalized or score < 0:
            continue
        parts.append(normalized)
        confidence_values.append(score)
    if not parts:
        return OcrExtraction(text="", confidence=0.0), raw_candidates
    extraction = OcrExtraction(
        text=normalize_ocr_text(" ".join(parts)),
        confidence=sum(confidence_values) / len(confidence_values),
    )
    return _postprocess_extraction(extraction), raw_candidates


def _postprocess_extraction(extraction: OcrExtraction) -> OcrExtraction:
    text = normalize_ocr_text(extraction.text)
    if _looks_like_noise(text):
        return OcrExtraction(text="", confidence=0.0)
    return OcrExtraction(text=text, confidence=extraction.confidence)


def _parse_paddle_result(result: object) -> OcrExtraction:
    texts: list[str] = []
    confidences: list[float] = []
    if not isinstance(result, list):
        return OcrExtraction(text="", confidence=0.0)

    def walk(node: object) -> None:
        if isinstance(node, dict):
            rec_texts = node.get("rec_texts")
            rec_scores = node.get("rec_scores")
            if isinstance(rec_texts, list):
                for index, text in enumerate(rec_texts):
                    if isinstance(text, str):
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
    original = normalized
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
        before = normalized
        normalized = normalized.replace(source, target)
        if normalized != before:
            _LOG.debug("      norm: %r → %r  (replaced %r→%r)",
                       _repr_text(before), _repr_text(normalized), source, target)
    before = normalized
    normalized = normalized.replace("°/o", "%")
    if normalized != before:
        _LOG.debug("      norm: degree/o → %%")
    before = normalized
    for pattern in PLUS_MINUS_PATTERNS:
        normalized = pattern.sub("\u00b1", normalized)
    # Also convert trailing + after digits — OCR sometimes reads "86±" as "86+"
    # when ± is at the end of the visible area (no digit follows in the crop).
    normalized = re.sub(r"(?<=\d)\+$", "\u00b1", normalized)
    if normalized != before:
        _LOG.debug("      norm: pm_pattern: %r → %r", _repr_text(before), _repr_text(normalized))
    normalized = re.sub(r"\s*%\s*", "%", normalized)
    normalized = re.sub(r"\s*\u00b1\s*", "\u00b1", normalized)
    before = normalized
    # Space between digits in dimension context means OCR dropped ± glyph:
    # "86 10%kg" → "86±10%kg", "296 5" → "296±5"
    # Tolerance is usually 1-2 digits (or 1-3 before %)
    normalized = re.sub(r"(?<=\d) (?=\d{1,2}(?:[^\d]|$))", "\u00b1", normalized)
    normalized = re.sub(r"(?<=\d) (?=\d{1,3}%)", "\u00b1", normalized)
    if normalized != before:
        _LOG.debug("      norm: space→± : %r → %r", _repr_text(before), _repr_text(normalized))
    normalized = normalized.replace(" ", "")
    if normalized != original:
        _LOG.debug("      norm: final: %r → %r", _repr_text(original), _repr_text(normalized))
    return normalized


def _looks_like_noise(text: str) -> bool:
    if not text:
        return True
    if NOISE_ONLY_PATTERN.match(text):
        return True
    if len(text) <= 2 and not any(char.isalnum() for char in text):
        return True
    return False


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

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.fft import dctn

try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
except ImportError:
    pytesseract = None
    TesseractOutput = None


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}
TOKEN_PATTERN = re.compile(r"[A-ZА-Я0-9]{2,}")


@dataclass(slots=True)
class OcrConfig:
    mode: str = "auto"
    tesseract_cmd: str | None = None
    languages: str = "eng+rus"
    psm: int = 6


@dataclass(slots=True)
class ImageFeatures:
    path: Path
    phash: np.ndarray
    dhash: np.ndarray
    edge_map: np.ndarray
    projection_x: np.ndarray
    projection_y: np.ndarray
    fill_ratio: float
    content_bbox: np.ndarray
    occupancy_grid: np.ndarray
    component_count: int
    ocr_tokens: frozenset[str]
    ocr_bands: np.ndarray
    ocr_enabled: bool


def load_features(path: Path, ocr_config: OcrConfig | None = None) -> ImageFeatures:
    image = Image.open(path).convert("L")
    normalized = _normalize_image(image)
    binary = _binary_map(normalized)
    phash = _phash(normalized)
    dhash = _dhash(normalized)
    edge_map = _edge_map(normalized)
    projection_x = edge_map.mean(axis=0)
    projection_y = edge_map.mean(axis=1)
    fill_ratio = float(binary.mean())
    content_bbox = _content_bbox(binary)
    occupancy_grid = _occupancy_grid(binary)
    component_count = _component_count(binary)
    ocr_tokens, ocr_bands, ocr_enabled = _extract_ocr_features(image, ocr_config)
    return ImageFeatures(
        path=path,
        phash=phash,
        dhash=dhash,
        edge_map=edge_map,
        projection_x=projection_x,
        projection_y=projection_y,
        fill_ratio=fill_ratio,
        content_bbox=content_bbox,
        occupancy_grid=occupancy_grid,
        component_count=component_count,
        ocr_tokens=ocr_tokens,
        ocr_bands=ocr_bands,
        ocr_enabled=ocr_enabled,
    )


def _normalize_image(image: Image.Image, size: int = 256) -> np.ndarray:
    image = ImageOps.autocontrast(image)
    image = ImageOps.pad(image, (size, size), color=255, method=Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def _binary_map(image: np.ndarray) -> np.ndarray:
    threshold = float(np.percentile(image, 82))
    binary = image < min(threshold, 0.92)
    cleaned = ndimage.binary_opening(binary, structure=np.ones((2, 2), dtype=bool))
    return cleaned.astype(np.uint8)


def _phash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    size = hash_size * highfreq_factor
    pil = Image.fromarray((image * 255).astype(np.uint8), mode="L").resize((size, size), Image.Resampling.LANCZOS)
    data = np.asarray(pil, dtype=np.float32)
    coeffs = dctn(data, norm="ortho")
    low_freq = coeffs[:hash_size, :hash_size]
    median = np.median(low_freq[1:, 1:])
    return (low_freq > median).astype(np.uint8).reshape(-1)


def _dhash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    pil = Image.fromarray((image * 255).astype(np.uint8), mode="L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    data = np.asarray(pil, dtype=np.float32)
    return (data[:, 1:] > data[:, :-1]).astype(np.uint8).reshape(-1)


def _edge_map(image: np.ndarray, size: int = 64) -> np.ndarray:
    pil = Image.fromarray((image * 255).astype(np.uint8), mode="L").resize((size, size), Image.Resampling.LANCZOS)
    data = np.asarray(pil, dtype=np.float32) / 255.0
    gx = np.zeros_like(data)
    gy = np.zeros_like(data)
    gx[:, 1:-1] = data[:, 2:] - data[:, :-2]
    gy[1:-1, :] = data[2:, :] - data[:-2, :]
    magnitude = np.hypot(gx, gy)
    threshold = float(np.percentile(magnitude, 72))
    return (magnitude >= threshold).astype(np.uint8)


def _content_bbox(binary: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(binary)
    if len(xs) == 0 or len(ys) == 0:
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    height, width = binary.shape
    return np.array(
        [
            xs.min() / width,
            ys.min() / height,
            xs.max() / width,
            ys.max() / height,
        ],
        dtype=np.float32,
    )


def _occupancy_grid(binary: np.ndarray, grid_size: int = 4) -> np.ndarray:
    height, width = binary.shape
    block_h = height // grid_size
    block_w = width // grid_size
    values: list[float] = []
    for row in range(grid_size):
        for col in range(grid_size):
            top = row * block_h
            left = col * block_w
            bottom = height if row == grid_size - 1 else (row + 1) * block_h
            right = width if col == grid_size - 1 else (col + 1) * block_w
            values.append(float(binary[top:bottom, left:right].mean()))
    return np.array(values, dtype=np.float32)


def _component_count(binary: np.ndarray) -> int:
    labeled, count = ndimage.label(binary)
    if count == 0:
        return 0
    sizes = ndimage.sum(binary, labeled, range(1, count + 1))
    significant = np.count_nonzero(np.asarray(sizes) >= 12)
    return int(significant)


def _extract_ocr_features(image: Image.Image, ocr_config: OcrConfig | None) -> tuple[frozenset[str], np.ndarray, bool]:
    config = ocr_config or OcrConfig()
    if config.mode == "off":
        return frozenset(), np.zeros(6, dtype=np.float32), False
    ocr_ready = _is_ocr_available(config)
    if not ocr_ready:
        if config.mode == "required":
            raise RuntimeError("OCR mode is required, but pytesseract or tesseract executable is not available.")
        return frozenset(), np.zeros(6, dtype=np.float32), False

    assert pytesseract is not None
    if config.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

    prepared = image.convert("L")
    prepared = ImageOps.autocontrast(prepared)
    prepared = prepared.resize((prepared.width * 2, prepared.height * 2), Image.Resampling.LANCZOS)
    thresholded = prepared.point(lambda px: 0 if px < 210 else 255, mode="1")
    custom = f"--oem 3 --psm {config.psm}"

    try:
        data = pytesseract.image_to_data(
            thresholded,
            lang=config.languages,
            config=custom,
            output_type=TesseractOutput.DICT,
        )
    except Exception:
        if config.mode == "required":
            raise
        return frozenset(), np.zeros(6, dtype=np.float32), False

    tokens: set[str] = set()
    bands = np.zeros(6, dtype=np.float32)
    total = max(1, thresholded.height)
    texts = data.get("text", [])
    tops = data.get("top", [])
    heights = data.get("height", [])
    confidences = data.get("conf", [])

    for text, top, box_height, confidence in zip(texts, tops, heights, confidences):
        if str(confidence).strip() in {"", "-1"}:
            continue
        try:
            if float(confidence) < 20.0:
                continue
        except ValueError:
            continue
        normalized = _normalize_tokens(str(text))
        if not normalized:
            continue
        tokens.update(normalized)
        center = (int(top) + int(box_height) * 0.5) / total
        band_index = min(len(bands) - 1, max(0, int(center * len(bands))))
        bands[band_index] += len(normalized)

    if bands.sum() > 0:
        bands /= bands.sum()
    return frozenset(sorted(tokens)), bands, True


def _normalize_tokens(text: str) -> list[str]:
    upper = text.upper().replace("O", "0")
    return TOKEN_PATTERN.findall(upper)


def _is_ocr_available(config: OcrConfig) -> bool:
    if pytesseract is None:
        return False
    if config.tesseract_cmd:
        return Path(config.tesseract_cmd).exists()
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return True
    return shutil.which("tesseract") is not None


def hash_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return 100.0 * (1.0 - np.count_nonzero(left != right) / left.size)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 and right_norm == 0.0:
        return 100.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return max(0.0, min(100.0, similarity * 100.0))


def edge_similarity(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 100.0
    return 100.0 * float(intersection / union)


def ratio_similarity(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), 1e-6)
    return max(0.0, 100.0 * (1.0 - abs(left - right) / scale))


def bbox_similarity(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.abs(left - right).mean()
    return max(0.0, 100.0 * (1.0 - float(delta)))


def token_similarity(left: frozenset[str], right: frozenset[str]) -> float | None:
    if not left and not right:
        return None
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    if union == 0:
        return None
    return 100.0 * intersection / union


def structure_similarity(left: ImageFeatures, right: ImageFeatures) -> float:
    return round(
        (
            ratio_similarity(left.fill_ratio, right.fill_ratio) * 0.20
            + bbox_similarity(left.content_bbox, right.content_bbox) * 0.25
            + cosine_similarity(left.occupancy_grid, right.occupancy_grid) * 0.35
            + ratio_similarity(float(left.component_count), float(right.component_count)) * 0.20
        ),
        2,
    )


def ocr_similarity(left: ImageFeatures, right: ImageFeatures) -> float | None:
    token_score = token_similarity(left.ocr_tokens, right.ocr_tokens)
    band_score = cosine_similarity(left.ocr_bands, right.ocr_bands)
    if token_score is None:
        if not left.ocr_enabled and not right.ocr_enabled:
            return None
        return 0.0
    return round(token_score * 0.80 + band_score * 0.20, 2)


def combined_similarity(left: ImageFeatures, right: ImageFeatures) -> float:
    components: list[tuple[float, float]] = []
    components.append((hash_similarity(left.phash, right.phash), 0.12))
    components.append((hash_similarity(left.dhash, right.dhash), 0.08))
    components.append((edge_similarity(left.edge_map, right.edge_map), 0.22))
    projection_score = (cosine_similarity(left.projection_x, right.projection_x) + cosine_similarity(left.projection_y, right.projection_y)) / 2.0
    components.append((projection_score, 0.12))
    components.append((structure_similarity(left, right), 0.26))

    ocr_score = ocr_similarity(left, right)
    if ocr_score is not None:
        components.append((ocr_score, 0.20))

    weighted_sum = sum(score * weight for score, weight in components)
    total_weight = sum(weight for _, weight in components)
    return round(weighted_sum / total_weight, 2)

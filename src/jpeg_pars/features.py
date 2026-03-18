from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}


@dataclass(slots=True)
class ImageFeatures:
    path: Path
    phash: np.ndarray
    dhash: np.ndarray
    edge_map: np.ndarray
    projection_x: np.ndarray
    projection_y: np.ndarray


def load_features(path: Path) -> ImageFeatures:
    image = Image.open(path).convert("L")
    normalized = _normalize_image(image)
    phash = _phash(normalized)
    dhash = _dhash(normalized)
    edge_map = _edge_map(normalized)
    projection_x = edge_map.mean(axis=0)
    projection_y = edge_map.mean(axis=1)
    return ImageFeatures(
        path=path,
        phash=phash,
        dhash=dhash,
        edge_map=edge_map,
        projection_x=projection_x,
        projection_y=projection_y,
    )


def _normalize_image(image: Image.Image, size: int = 256) -> np.ndarray:
    image = ImageOps.autocontrast(image)
    image = ImageOps.pad(image, (size, size), color=255, method=Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


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


def combined_similarity(left: ImageFeatures, right: ImageFeatures) -> float:
    phash_score = hash_similarity(left.phash, right.phash)
    dhash_score = hash_similarity(left.dhash, right.dhash)
    edge_score = edge_similarity(left.edge_map, right.edge_map)
    projection_score = (cosine_similarity(left.projection_x, right.projection_x) + cosine_similarity(left.projection_y, right.projection_y)) / 2.0
    score = (
        phash_score * 0.30
        + dhash_score * 0.20
        + edge_score * 0.35
        + projection_score * 0.15
    )
    return round(score, 2)

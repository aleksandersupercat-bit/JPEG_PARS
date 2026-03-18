from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .features import SUPPORTED_EXTENSIONS, ImageFeatures, combined_similarity, load_features


@dataclass(slots=True)
class ClusterResult:
    groups: list[list[Path]]
    scores: dict[str, dict[str, float]]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def discover_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def cluster_images(input_dir: Path, similarity_threshold: int, recursive: bool = False) -> ClusterResult:
    files = discover_images(input_dir, recursive=recursive)
    if not files:
        raise ValueError(f"No JPEG files found in {input_dir}")

    features = [load_features(path) for path in files]
    groups = _build_groups(features, similarity_threshold)
    scores = _build_score_map(features, groups)
    return ClusterResult(groups=groups, scores=scores)


def _build_groups(features: list[ImageFeatures], similarity_threshold: int) -> list[list[Path]]:
    union_find = UnionFind(len(features))
    for left_index in range(len(features)):
        for right_index in range(left_index + 1, len(features)):
            score = combined_similarity(features[left_index], features[right_index])
            if score >= similarity_threshold:
                union_find.union(left_index, right_index)

    grouped: dict[int, list[Path]] = {}
    for index, item in enumerate(features):
        root = union_find.find(index)
        grouped.setdefault(root, []).append(item.path)

    return sorted(
        (sorted(paths) for paths in grouped.values()),
        key=lambda paths: (-len(paths), str(paths[0]).lower()),
    )


def _build_score_map(features: list[ImageFeatures], groups: list[list[Path]]) -> dict[str, dict[str, float]]:
    feature_by_path = {feature.path: feature for feature in features}
    result: dict[str, dict[str, float]] = {}
    for group in groups:
        if len(group) < 2:
            continue
        anchor = group[0]
        result[str(anchor)] = {}
        for path in group[1:]:
            result[str(anchor)][str(path)] = combined_similarity(feature_by_path[anchor], feature_by_path[path])
    return result


def materialize_groups(
    result: ClusterResult,
    output_dir: Path,
    source_root: Path,
    mode: str = "copy",
    min_group_size: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, group in enumerate(result.groups, start=1):
        if len(group) < min_group_size:
            continue
        group_dir = output_dir / f"group_{index:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        for item in group:
            destination = group_dir / item.name
            if destination.exists():
                destination = group_dir / f"{item.stem}_{abs(hash(item)) % 100000}{item.suffix.lower()}"
            if mode == "move":
                shutil.move(str(item), str(destination))
            else:
                shutil.copy2(item, destination)

    report_path = output_dir / "report.json"
    report = {
        "source_root": str(source_root),
        "group_count": len(result.groups),
        "groups": [
            {
                "group": f"group_{index:03d}",
                "size": len(group),
                "files": [str(path) for path in group],
            }
            for index, group in enumerate(result.groups, start=1)
            if len(group) >= min_group_size
        ],
        "scores_from_anchor": result.scores,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

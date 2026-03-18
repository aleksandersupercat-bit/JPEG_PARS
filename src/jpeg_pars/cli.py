from __future__ import annotations

import argparse
from pathlib import Path

from .clustering import cluster_images, materialize_groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jpeg-pars",
        description="Sort JPEG drawings into folders based on visual similarity.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Folder with JPEG images.")
    parser.add_argument("--output", required=True, type=Path, help="Folder where grouped files will be written.")
    parser.add_argument(
        "--similarity",
        required=True,
        type=int,
        help="Similarity threshold from 1 to 100.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="Copy files by default; move if requested.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search JPEG files in nested folders.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=1,
        help="Skip writing groups smaller than this value.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists() or not args.input.is_dir():
        parser.error(f"Input folder does not exist: {args.input}")

    if not 1 <= args.similarity <= 100:
        parser.error("--similarity must be in range 1..100")

    if args.min_group_size < 1:
        parser.error("--min-group-size must be >= 1")

    result = cluster_images(
        input_dir=args.input,
        similarity_threshold=args.similarity,
        recursive=args.recursive,
    )
    materialize_groups(
        result=result,
        output_dir=args.output,
        source_root=args.input,
        mode=args.mode,
        min_group_size=args.min_group_size,
    )

    file_count = sum(len(group) for group in result.groups)
    print(f"Processed {file_count} files into {len(result.groups)} groups. Output: {args.output}")


if __name__ == "__main__":
    main()

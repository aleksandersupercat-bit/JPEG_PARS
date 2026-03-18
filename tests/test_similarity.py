import tempfile
import unittest
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jpeg_pars.clustering import cluster_images
from jpeg_pars.features import OcrConfig


def _make_test_image(path: Path, variant: str) -> None:
    image = Image.new("L", (400, 300), color=255)
    draw = ImageDraw.Draw(image)
    if variant == "base":
        draw.rectangle((40, 40, 340, 240), outline=0, width=4)
        draw.line((40, 140, 340, 140), fill=0, width=3)
    elif variant == "close":
        draw.rectangle((50, 40, 350, 240), outline=0, width=4)
        draw.line((50, 145, 350, 145), fill=0, width=3)
    else:
        draw.ellipse((90, 50, 300, 250), outline=0, width=5)
    image.save(path, quality=95)


class ClusterImagesTests(unittest.TestCase):
    def test_groups_similar_drawings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _make_test_image(tmp_path / "a.jpg", "base")
            _make_test_image(tmp_path / "b.jpg", "close")
            _make_test_image(tmp_path / "c.jpg", "other")

            result = cluster_images(
                tmp_path,
                similarity_threshold=70,
                ocr_config=OcrConfig(mode="off"),
            )

            sizes = sorted(len(group) for group in result.groups)
            self.assertEqual(sizes, [1, 2])

    def test_ocr_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _make_test_image(tmp_path / "a.jpg", "base")
            _make_test_image(tmp_path / "b.jpg", "close")

            result = cluster_images(
                tmp_path,
                similarity_threshold=70,
                ocr_config=OcrConfig(mode="off"),
            )

            self.assertFalse(result.ocr_enabled)
            self.assertEqual(len(result.groups), 1)


if __name__ == "__main__":
    unittest.main()

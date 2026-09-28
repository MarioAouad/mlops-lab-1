import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.food11.data import CATEGORIES, SPLITS, prepare_dataset


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for split in SPLITS:
            path = self.root / "BigData/food11_raw" / split
            path.mkdir(parents=True)
            for label in range(len(CATEGORIES)):
                Image.new("RGB", (20, 30)).save(path / f"{label}_0.jpg")

    def tearDown(self):
        self.temp.cleanup()

    def test_preparation_creates_the_expected_mapping(self):
        prepare_dataset(self.root)
        for dataset in ["food11_processed", "food11_processed_mini"]:
            for split in SPLITS:
                for label, category in enumerate(CATEGORIES):
                    with Image.open(self.root / "BigData" / dataset / split / category / f"{label}_0.jpg") as image:
                        self.assertEqual(image.size, (128, 128))
                        self.assertEqual(image.mode, "RGB")

    def test_existing_data_is_preserved_by_default(self):
        output = self.root / "BigData/food11_processed"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("original")
        with self.assertRaises(FileExistsError):
            prepare_dataset(self.root)
        self.assertEqual(marker.read_text(), "original")

    def test_overwrite_is_explicit(self):
        prepare_dataset(self.root)
        marker = self.root / "BigData/food11_processed/obsolete.txt"
        marker.write_text("old")
        prepare_dataset(self.root, overwrite=True)
        self.assertFalse(marker.exists())

    def test_invalid_input_does_not_delete_existing_outputs(self):
        output = self.root / "BigData/food11_processed"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("original")
        (self.root / "BigData/food11_raw/validation/0_0.jpg").unlink()
        with self.assertRaises(ValueError):
            prepare_dataset(self.root, overwrite=True)
        self.assertTrue(marker.exists())

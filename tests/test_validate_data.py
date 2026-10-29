import subprocess
import sys
from pathlib import Path
from shutil import copyfile

import pytest
from PIL import Image

from src.food11.data import CLASSES, SPLITS
from src.food11.validate_data import validate_dataset


@pytest.fixture
def dataset(tmp_path):
    for split_index, split in enumerate(SPLITS):
        for class_index, category in enumerate(CLASSES):
            folder = tmp_path / split / category
            folder.mkdir(parents=True)
            Image.new("RGB", (128, 128), (split_index * 80, class_index * 20, 17)).save(folder / "image.png")
    return tmp_path


def test_valid_dataset_passes(dataset):
    assert all(passed for _, passed, _ in validate_dataset(dataset, min_per_class=1))


def test_duplicate_fails_and_cli_exits_nonzero(dataset):
    copyfile(dataset / "training/Bread/image.png", dataset / "evaluation/Bread/copy.png")
    checks = {name: passed for name, passed, _ in validate_dataset(dataset, min_per_class=1)}
    assert not checks["Cross-split duplicates"]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "src/food11/validate_data.py"), str(dataset), "--min-per-class", "1"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "FAIL Cross-split duplicates" in result.stdout


def test_missing_class_and_insufficient_counts_fail(dataset):
    folder = dataset / "validation/Bread"
    (folder / "image.png").unlink()
    folder.rmdir()
    checks = {name: passed for name, passed, _ in validate_dataset(dataset, min_per_class=1)}
    assert not checks["Class folders"]
    assert not checks["Minimum counts"]
    assert not checks["Class balance"]


@pytest.mark.parametrize("mode,size", [("L", (128, 128)), ("RGB", (64, 128))])
def test_wrong_image_format_fails(dataset, mode, size):
    Image.new(mode, size).save(dataset / "training/Bread/image.png")
    checks = {name: passed for name, passed, _ in validate_dataset(dataset, min_per_class=1)}
    assert not checks["Image format"]


def test_corrupt_image_fails(dataset):
    (dataset / "training/Bread/image.png").write_bytes(b"not an image")
    checks = {name: passed for name, passed, _ in validate_dataset(dataset, min_per_class=1)}
    assert not checks["Image format"]

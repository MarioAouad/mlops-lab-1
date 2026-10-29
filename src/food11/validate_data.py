"""Check the Food-11 processed dataset contract without changing its files."""

import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image

if __package__:
    from .data import CLASSES, IMAGE_EXTENSIONS, SPLITS
else:
    from data import CLASSES, IMAGE_EXTENSIONS, SPLITS


def validate_dataset(
    root: Path, min_per_class: int = 50, tau: float = 0.05
) -> list[tuple[str, bool, str]]:
    if min_per_class < 1 or not 0 <= tau <= 1:
        raise ValueError("min-per-class must be positive and tau must be between 0 and 1")
    folder_errors = []
    image_errors = []
    count_errors = []
    duplicates = []
    seen = {}
    counts = {}
    total = 0
    for split in SPLITS:
        split_root = root / split
        actual = {p.name for p in split_root.iterdir() if p.is_dir()} if split_root.is_dir() else set()
        if actual != set(CLASSES):
            folder_errors.append(
                f"{split}: missing={sorted(set(CLASSES) - actual)}, extra={sorted(actual - set(CLASSES))}"
            )
        counts[split] = {}
        for category in CLASSES:
            category_root = split_root / category
            files = sorted(
                p for p in category_root.glob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            counts[split][category] = len(files)
            if len(files) < min_per_class:
                count_errors.append(f"{split}/{category}: {len(files)} < {min_per_class}")
            for path in files:
                total += 1
                try:
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGB" or image.size != (128, 128):
                            image_errors.append(f"{path}: {image.mode}, {image.size}")
                    digest = hashlib.md5(path.read_bytes()).hexdigest()
                    previous = seen.setdefault(digest, {})
                    for other_split, other_path in previous.items():
                        if other_split != split:
                            duplicates.append(f"{other_path} == {path}")
                    previous.setdefault(split, path)
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    image_errors.append(f"{path}: {exc}")

    balance_errors = []
    train_total = sum(counts["training"].values())
    validation_total = sum(counts["validation"].values())
    if not train_total or not validation_total:
        balance_errors.append("Training and validation splits must not be empty")
    else:
        for category in CLASSES:
            difference = abs(
                counts["training"][category] / train_total
                - counts["validation"][category] / validation_total
            )
            if difference > tau:
                balance_errors.append(f"{category}: share difference {difference:.6f} > {tau}")

    def result(name, errors, success):
        detail = f"{len(errors)} issue(s): " + "; ".join(str(e) for e in errors[:3]) if errors else success
        return name, not errors, detail

    return [
        result("Class folders", folder_errors, "Exactly 11 classes in each split"),
        result("Image format", image_errors, f"{total} readable RGB images at 128x128"),
        result("Minimum counts", count_errors, f"At least {min_per_class} images per class per split"),
        result("Cross-split duplicates", duplicates, "No identical file content across splits"),
        result("Class balance", balance_errors, f"Training/validation share differences <= {tau}"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()
    try:
        results = validate_dataset(args.dataset, args.min_per_class, args.tau)
    except ValueError as exc:
        parser.error(str(exc))
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
    return 0 if all(passed for _, passed, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())

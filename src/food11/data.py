import argparse
from pathlib import Path
import shutil

from PIL import Image


CATEGORIES = (
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
)
SPLITS = ("training", "evaluation", "validation")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100


def prepare_dataset(project_root: Path, *, overwrite: bool = False) -> None:
    project_root = project_root.resolve()
    raw_root = project_root / "BigData" / "food11_raw"
    processed_root = project_root / "BigData" / "food11_processed"
    mini_root = project_root / "BigData" / "food11_processed_mini"

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_root}")

    files_by_split = {}
    for split in SPLITS:
        split_root = raw_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing dataset split: {split_root}")

        files_by_category = {category: [] for category in CATEGORIES}
        for source in sorted(split_root.iterdir()):
            if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = source.stem.split("_", 1)[0]
            if not label.isdigit() or int(label) not in range(len(CATEGORIES)):
                raise ValueError(f"Could not determine a Food-11 label from {source.name}")
            files_by_category[CATEGORIES[int(label)]].append(source)

        if any(not sources for sources in files_by_category.values()):
            raise ValueError(f"Dataset split {split} must contain all eleven categories")
        files_by_split[split] = files_by_category

    # Validate all input splits before touching an existing output directory.
    for output_root in (processed_root, mini_root):
        if output_root.is_symlink() or output_root.resolve().parent != project_root / "BigData":
            raise ValueError(f"Unsafe dataset output path: {output_root}")
        if output_root.exists() and not overwrite:
            raise FileExistsError(f"Dataset already exists: {output_root}. Use --overwrite to replace it.")
    for output_root in (processed_root, mini_root):
        if output_root.exists():
            shutil.rmtree(output_root)

    for split, files_by_category in files_by_split.items():
        for category, sources in files_by_category.items():
            processed_category = processed_root / split / category
            mini_category = mini_root / split / category
            processed_category.mkdir(parents=True, exist_ok=True)
            mini_category.mkdir(parents=True, exist_ok=True)

            for index, source in enumerate(sources):
                destination = processed_category / source.name
                with Image.open(source) as image:
                    image.convert("RGB").resize(IMAGE_SIZE).save(destination, format="JPEG")

                if index < MINI_LIMIT:
                    shutil.copy2(destination, mini_category / source.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Food-11 images from the DVC-tracked raw data.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing processed datasets")
    args = parser.parse_args()
    prepare_dataset(Path(__file__).resolve().parents[2], overwrite=args.overwrite)

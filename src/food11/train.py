import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

from .data import CATEGORIES


NUM_CLASSES = 11
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOTS = {
    "processed": PROJECT_ROOT / "BigData" / "food11_processed",
    "mini": PROJECT_ROOT / "BigData" / "food11_processed_mini",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet18 on Food-11.")
    parser.add_argument("--dataset", choices=DATASET_ROOTS, default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    return parser.parse_args()


def build_dataloaders(dataset_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    weights = ResNet18_Weights.DEFAULT
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])

    datasets_by_split = {
        split: datasets.ImageFolder(dataset_root / split, transform=image_transform)
        for split in ("training", "validation", "evaluation")
    }

    for split, dataset in datasets_by_split.items():
        if dataset.classes != list(CATEGORIES):
            raise ValueError(f"Incorrect Food-11 class mapping in {split}: {dataset.classes}")

    return tuple(
        DataLoader(
            datasets_by_split[split],
            batch_size=batch_size,
            shuffle=split == "training",
            num_workers=0,
        )
        for split in ("training", "validation", "evaluation")
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dataset_root = DATASET_ROOTS[args.dataset]
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("food11")

    train_loader, validation_loader, test_loader = build_dataloaders(
        dataset_root, args.batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
                "device": str(device),
                "seed": args.seed,
            }
        )

        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                check=True, capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unavailable"
        mlflow.set_tag("git_commit", commit)

        for epoch in range(args.epochs):
            train_loss, _ = run_epoch(
                model, train_loader, loss_function, device, optimizer
            )
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, loss_function, device
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", validation_loss, step=epoch)
            mlflow.log_metric("val_accuracy", validation_accuracy, step=epoch)
            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={validation_loss:.4f}, "
                f"val_accuracy={validation_accuracy:.4f}"
            )

        _, test_accuracy = run_epoch(model, test_loader, loss_function, device)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.pytorch.log_model(
            model, name="model", serialization_format="pickle",
            metadata={"class_names": list(CATEGORIES)},
        )
        print(f"test_accuracy={test_accuracy:.4f}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


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

    if len(datasets_by_split["training"].classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, found {len(datasets_by_split['training'].classes)}"
        )

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
    args = parse_args()
    dataset_root = DATASET_ROOTS[args.dataset]
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    mlflow.set_tracking_uri("http://127.0.0.1:5001")
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
            }
        )

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
        mlflow.pytorch.log_model(model, "model", serialization_format="pickle")
        print(f"test_accuracy={test_accuracy:.4f}")


if __name__ == "__main__":
    main()

"""Apply the Lab 5 accuracy gate before moving the food11 champion alias."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from mlflow.exceptions import MlflowException

import mlflow


def accuracy_threshold(accuracy: float, test_size: int) -> tuple[float, float]:
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Accuracy must be a finite number between 0 and 1")
    if test_size < 1:
        raise ValueError("Test-set size must be positive")
    standard_error = math.sqrt(accuracy * (1 - accuracy) / test_size)
    return standard_error, accuracy + 2 * standard_error


def run_accuracy(client, version, metric: str) -> float:
    if not version.run_id:
        raise ValueError(f"Model version {version.version} has no producing run ID")
    metrics = client.get_run(version.run_id).data.metrics
    if metric not in metrics:
        raise ValueError(f"Run {version.run_id} has no {metric} metric")
    accuracy = float(metrics[metric])
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError(f"Run {version.run_id} has an invalid {metric} metric")
    return accuracy


def promote_version(client, version: str, test_size: int, name: str = "food11", metric: str = "test_accuracy") -> dict:
    candidate = client.get_model_version(name, version)
    champion = client.get_model_version_by_alias(name, "champion")
    candidate_accuracy = run_accuracy(client, candidate, metric)
    champion_accuracy = run_accuracy(client, champion, metric)
    standard_error, threshold = accuracy_threshold(champion_accuracy, test_size)
    promoted = candidate_accuracy >= threshold
    decision = "promoted" if promoted else "challenger"
    alias = "champion" if promoted else "challenger"
    client.set_registered_model_alias(name, alias, candidate.version)
    report = {
        "decision": decision,
        "candidate_version": str(candidate.version),
        "champion_version_before": str(champion.version),
        "candidate_accuracy": candidate_accuracy,
        "champion_accuracy": champion_accuracy,
        "test_size": test_size,
        "standard_error": standard_error,
        "two_standard_errors": 2 * standard_error,
        "threshold": threshold,
        "metric": metric,
    }
    for key, value in report.items():
        client.set_model_version_tag(name, candidate.version, f"promotion.{key}", str(value))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--name", default="food11")
    parser.add_argument("--metric", default="test_accuracy")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    parser.add_argument("--test-size", type=int, help="Number of images in the evaluation split")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[2] / "BigData/food11_processed_mini")
    args = parser.parse_args()
    if args.version < 1:
        parser.error("version must be positive")
    test_size = args.test_size
    if test_size is None:
        evaluation = args.dataset / "evaluation"
        if not evaluation.is_dir():
            parser.error("Evaluation split is missing; supply --dataset or --test-size")
        test_size = sum(1 for path in evaluation.glob("*/*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if test_size < 1:
        parser.error("test-size must be positive")
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri)
    try:
        report = promote_version(client, str(args.version), test_size, args.name, args.metric)
    except (MlflowException, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

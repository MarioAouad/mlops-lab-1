"""Import an existing local MLflow model into a fresh tracking server."""

import argparse
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import Model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_local_champion(project_root: Path = PROJECT_ROOT) -> Path:
    database = project_root / "mlflow.db"
    if not database.is_file():
        raise FileNotFoundError("No local MLflow database. Supply --source with a model artifact directory.")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        row = connection.execute(
            "SELECT v.storage_location FROM model_versions v "
            "JOIN registered_model_aliases a ON a.name=v.name AND a.version=v.version "
            "WHERE a.name='food11' AND a.alias='champion'"
        ).fetchone()
    if row is None:
        raise ValueError("The local registry has no food11@champion alias. Supply --source.")
    uri = urlparse(row[0])
    if uri.scheme == "mlflow-artifacts":
        source = project_root / "mlruns" / unquote(uri.path).lstrip("/")
    elif uri.scheme == "file":
        source = Path(url2pathname(uri.path))
    else:
        raise ValueError(f"Cannot locate local artifacts for URI scheme {uri.scheme!r}. Supply --source.")
    source = source.resolve()
    if not source.is_relative_to((project_root / "mlruns").resolve()):
        raise ValueError("The local model source is outside mlruns. Supply an explicit --source.")
    return source


def bootstrap(source: Path | None, tracking_uri: str, name: str = "food11", alias: str = "champion"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = mlflow.MlflowClient()
    try:
        existing = client.get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
    else:
        print(f"{name}@{alias} already points to version {existing.version}; no import needed.")
        return existing

    source = (source or find_local_champion()).resolve()
    if not (source / "MLmodel").is_file() or not (source / "data/model.pth").is_file():
        raise FileNotFoundError(f"Expected a complete PyTorch MLflow artifact directory: {source}")
    metadata = Model.load(str(source / "MLmodel"))
    if "pytorch" not in metadata.flavors:
        raise ValueError("The seed artifact must contain the PyTorch flavor")

    mlflow.set_experiment(name)
    with mlflow.start_run(run_name="import-existing-champion") as run:
        mlflow.set_tags({"imported_from_run": metadata.run_id or "unknown", "purpose": "lab4-bootstrap"})
        mlflow.log_artifacts(str(source), artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
    version = mlflow.register_model(model_uri, name)
    client.set_registered_model_alias(name, alias, version.version)
    print(f"Imported {source} as {name} version {version.version}; assigned @{alias}.")
    return version


def main():
    # MLflow prints Unicode run links; redirected Windows stdout can be cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Local directory containing MLmodel and data/model.pth")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    parser.add_argument("--name", default="food11")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()
    bootstrap(args.source, args.tracking_uri, args.name, args.alias)


if __name__ == "__main__":
    main()

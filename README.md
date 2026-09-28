# Food-11 MLOps lab

Train a ResNet18 food classifier, track experiments and model versions with MLflow,
and serve predictions through FastAPI and a Streamlit upload page.

## Services and ports

| Service | Address from Windows | Address inside Compose |
| --- | --- | --- |
| MLflow UI and tracking API | http://127.0.0.1:5001 | http://mlflow:5000 |
| Inference API | Not published | http://inference:8000 |
| Streamlit frontend | http://127.0.0.1:8501 | http://frontend:8501 |

Port 5000 is occupied on this Windows machine, so MLflow publishes host port 5001.
The inference image contains application code and dependencies; the model is fetched
from MLflow when it starts. Model selection uses `models:/food11@champion`, following
Lab 3's aliases rather than Lab 4's older `Staging` examples.

## Requirements

- Python 3.12 and uv 0.12.10.
- Docker Desktop running with Linux containers and Docker Compose.
- DVC for dataset recovery and local data versioning.

```powershell
uv sync --locked
docker compose version
```

The project selects CPU-only PyTorch builds for Windows and Linux. GPU training
requires selecting a matching CUDA index and regenerating the lockfile intentionally.

## First start with your existing trained model

Run these commands from the repository root:

```powershell
docker compose up -d --build --wait mlflow
uv run --locked python -m src.food11.bootstrap
docker compose up -d --build --wait
docker compose ps
```

The first command creates a named volume for MLflow's SQLite database and artifacts.
The bootstrap command reads the old local `mlflow.db` in read-only mode, finds the
existing champion's artifact directory under `mlruns/`, uploads that model to the new
local MLflow server, registers it, and assigns `champion`. It does not retrain or
modify the original database or model files. Versions belong to each registry, so
the imported model can have a different version number from the old registry.

Bootstrap is repeatable: if the target registry already has the requested alias,
it leaves that alias and model version alone.

The existing model's saved requirements include unused CUDA packages. MLflow may
warn that those packages are missing in the CPU image; loading and predictions
have been verified with the CPU PyTorch runtime. The original artifact is preserved.

Open http://127.0.0.1:8501 and upload an image. Confidence is the softmax probability
of the predicted class and stays between 0 and 1.

For an exported model on another machine, supply its artifact folder explicitly:

```powershell
uv run --locked python -m src.food11.bootstrap --source "C:\path\to\model\artifacts"
```

A fresh clone does not include trained model artifacts. Supply an exported model
or train and register a new one before starting inference.

## Daily use and model updates

```powershell
docker compose up -d --wait
docker compose logs --tail 100 inference
docker compose restart inference
docker compose down
```

After assigning `champion` to a new model version in MLflow, restart inference to
load that version. Refreshing the frontend alone does not reload the model.
`docker compose down` preserves the named volume. Adding `-v` deletes the Compose
volume and its registry, database, and artifacts; an empty registry needs bootstrap
again before inference can start. The original host `mlflow.db` and `mlruns/` remain
outside that volume.

MLflow health gates inference startup, and inference health gates the frontend.
Inference also retries temporary MLflow errors with a finite number of attempts.
A missing model or alias fails startup immediately, so bootstrap the registry first.

## Configuration

`.env.example` lists the Compose defaults. Optionally copy it to the ignored `.env`
and change the host ports or `MODEL_URI`. For example, an explicit model version is
`models:/food11/2`. Internal service addresses remain unchanged when host ports change.

Local training and serving read `MLFLOW_TRACKING_URI`, defaulting to
`http://127.0.0.1:5001`. Training additionally accepts `--tracking-uri`.

MLflow serves artifacts over HTTP from `/mlflow-data/mlruns`; inference does not
need access to that directory. Its host allowlist includes the Compose hostname.

## Dataset preparation and training

The original Lab 1 `data.dvc` contains a one-image exercise snapshot. The full raw
Food-11 data is separately tracked by `BigData/food11_raw.dvc` (16,643 images).
Processed datasets are generated and ignored by Git.

The new raw-data pointer has been tracked locally only and has not been pushed to
DagsHub, at the owner's request. The same 16,643 files were already published in
the historical DVC snapshot at commit `d341d3f5ecf4dbaa15ce6831a060306b52f2e6b6`.
Its manifest matches the current raw data; a sample download was checked against
the local image's hash. A fresh clone can recover that existing snapshot without
uploading any new data:

```powershell
dvc get https://github.com/MarioAouad/mlops-lab-1.git data/food11_raw --rev d341d3f5ecf4dbaa15ce6831a060306b52f2e6b6 --out BigData/food11_raw
dvc commit BigData/food11_raw.dvc
```

Use the recovery command when `BigData/food11_raw` is missing. This machine already
has the raw dataset. Recovery requires access to the existing DagsHub remote.
`dvc commit` populates the local cache for the new pointer; it does not upload data.
Until that new directory manifest is published separately, use this historical
recovery command rather than relying on a fresh clone's `dvc pull` for the raw data.

Once raw data is present:

```powershell
uv run --locked python -m src.food11.data
uv run --locked python -m src.food11.train --dataset mini --epochs 5 --lr 0.0001 --batch-size 32 --seed 42
```

Dataset preparation refuses to replace existing processed data. An intentional
regeneration requires `--overwrite`. Keep that flag out of routine startup commands.
Training validates the eleven class labels across every split, records its seed
and Git commit, and saves the class mapping in the model's metadata.

For a new training run, register its logged model as `food11`, assign `champion`,
and restart inference. Training does not automatically replace a deployed alias.

## Tests and repository layout

```powershell
uv run --locked python -m unittest discover -s tests -v
docker compose config --quiet
```

Tests cover HTTP predictions, probability normalization, invalid uploads and model
outputs, startup retries, readiness, bootstrap behavior, and dataset preservation.
GitHub Actions runs the tests and validates the Compose configuration. CI does not
download the dataset or require the local trained model.

```text
src/food11/            Data preparation, training, inference, model bootstrap
mlflow/Dockerfile      MLflow tracking server with artifact proxying
frontend/             Streamlit application and image
docker-compose.yml    Three services, health checks, persistent MLflow volume
BigData/*.dvc         Full raw dataset pointer
data.dvc              Original Lab 1 exercise pointer
tests/                Regression tests
lab/                  Historical lab answers
command.md            Windows commands and legacy Lab 3 workflow
```

The root Dockerfile is the main multi-stage inference image.
`Dockerfile.single-stage` remains available for the Lab 3 comparison. Both use the
same lockfile and virtual environment. Datasets, local environments, Git history,
and model artifacts are excluded from the inference build context.

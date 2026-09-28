import io
import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import load_model
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from .data import CATEGORIES

DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
DEFAULT_MODEL_URI = "models:/food11@champion"
CLASS_NAMES = list(CATEGORIES)
logger = logging.getLogger(__name__)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=ResNet18_Weights.DEFAULT.transforms().mean,
        std=ResNet18_Weights.DEFAULT.transforms().std,
    ),
])


def load_model_with_retry(model_uri: str, attempts: int, delay: float):
    """Retry temporary MLflow failures; report missing aliases without waiting."""
    if attempts < 1 or delay < 0:
        raise ValueError("Model load attempts must be positive and delay nonnegative")
    for attempt in range(1, attempts + 1):
        try:
            return load_model(model_uri)
        except MlflowException as exc:
            if exc.get_http_status_code() < 500 or attempt == attempts:
                raise
            logger.warning("MLflow model load failed (attempt %s/%s): %s", attempt, attempts, exc)
            time.sleep(delay)


@asynccontextmanager
async def lifespan(application: FastAPI):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    model_uri = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
    model = await run_in_threadpool(
        load_model_with_retry,
        model_uri,
        int(os.getenv("MODEL_LOAD_ATTEMPTS", "12")),
        float(os.getenv("MODEL_LOAD_RETRY_DELAY", "5")),
    )
    metadata = model.metadata.metadata or {}
    if metadata.get("class_names", CLASS_NAMES) != CLASS_NAMES:
        raise ValueError("The model's class mapping does not match Food-11")
    application.state.model = model
    application.state.model_uri = model_uri
    logger.info("Loaded %s", model_uri)
    try:
        yield
    finally:
        application.state.model = None


app = FastAPI(title="Food11 API", lifespan=lifespan)
app.state.model = None


@app.get("/health")
def health() -> dict[str, str]:
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


def classify(contents: bytes, model) -> dict[str, float | str]:
    try:
        with Image.open(io.BytesIO(contents)) as uploaded_image:
            image = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    tensor = image_transform(image)
    image_batch = np.asarray(tensor.unsqueeze(0).numpy(), dtype=np.float32)
    prediction = model.predict(image_batch)
    if hasattr(prediction, "values"):
        prediction = prediction.values
    logits = np.asarray(prediction, dtype=np.float64)
    if logits.shape == (1, len(CLASS_NAMES)):
        logits = logits[0]
    if logits.shape != (len(CLASS_NAMES),) or not np.isfinite(logits).all():
        raise HTTPException(status_code=500, detail="Model returned invalid class scores")

    # ResNet returns logits. Subtracting their maximum prevents exponential overflow.
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    predicted_index = int(np.argmax(probabilities))
    return {
        "category": CLASS_NAMES[predicted_index],
        "confidence": round(float(probabilities[predicted_index]), 6),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float | str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No image file supplied")
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return await run_in_threadpool(classify, await file.read(), model)

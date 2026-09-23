import io
import os
from pathlib import Path

import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_URI = "models:/food11@champion"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"

DATASET_DIR = PROJECT_ROOT / "BigData" / "food11_processed_mini" / "validation"
CLASS_NAMES = sorted(
    [path.name for path in DATASET_DIR.iterdir() if path.is_dir()]
) if DATASET_DIR.exists() else [
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
]

image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=ResNet18_Weights.DEFAULT.transforms().mean,
            std=ResNet18_Weights.DEFAULT.transforms().std,
        ),
    ]
)

app = FastAPI(title="Food11 API")
app.state.model = None


@app.on_event("startup")
def startup_event() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float | str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No image file supplied")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    tensor = image_transform(image)
    image_batch = np.asarray(tensor.unsqueeze(0).numpy(), dtype=np.float32)

    model = app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    prediction = model.predict(image_batch)
    if hasattr(prediction, "values"):
        prediction = prediction.values

    prediction = np.asarray(prediction)
    if prediction.ndim == 2 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim == 0:
        prediction = np.asarray([prediction])

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    if predicted_index >= len(CLASS_NAMES):
        category = str(predicted_index)
    else:
        category = CLASS_NAMES[predicted_index]

    return {"category": category, "confidence": round(confidence, 6)}

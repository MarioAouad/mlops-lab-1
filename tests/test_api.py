from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.food11 import serve


class FakeModel:
    def predict(self, x):
        # The serving code expects logits and applies softmax itself.
        return np.log([[0.90] + [0.01] * 10])


def test_health(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        assert client.get("/health").status_code == 200


def test_predict_returns_valid_json(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())
        fixture = Path(__file__).parent / "fixtures" / "bread.jpg"
        with fixture.open("rb") as f:
            response = client.post(
                "/predict", files={"file": ("bread.jpg", f, "image/jpeg")}
            )
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Bread"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["confidence"] == 0.90

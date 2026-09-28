import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST
from PIL import Image

from src.food11 import serve


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.metadata = SimpleNamespace(metadata={})
        self.model.predict.return_value = np.zeros((1, 11), dtype=np.float32)
        image = io.BytesIO()
        Image.new("RGB", (16, 16)).save(image, format="PNG")
        self.image = image.getvalue()
        self.loader = patch.object(serve, "load_model", return_value=self.model)
        self.loader.start()
        self.client = TestClient(serve.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.loader.stop()

    def predict(self):
        return self.client.post("/predict", files={"file": ("food.png", self.image, "image/png")})

    def test_confidence_is_probability_for_equal_logits(self):
        response = self.predict()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["category"], "Bread")
        self.assertAlmostEqual(response.json()["confidence"], 1 / 11, places=6)
        batch = self.model.predict.call_args.args[0]
        self.assertEqual(batch.shape, (1, 3, 224, 224))
        self.assertEqual(batch.dtype, np.float32)

    def test_softmax_is_stable_for_large_logits(self):
        self.model.predict.return_value = np.array([[10000, 10002] + [9990] * 9])
        result = self.predict().json()
        self.assertEqual(result["category"], "Dairy product")
        self.assertTrue(0.88 < result["confidence"] < 0.89)

    def test_invalid_image_returns_400(self):
        response = self.client.post("/predict", files={"file": ("bad.jpg", b"bad image", "image/jpeg")})
        self.assertEqual(response.status_code, 400)
        self.model.predict.assert_not_called()

    def test_missing_upload_returns_422(self):
        self.assertEqual(self.client.post("/predict").status_code, 422)

    def test_readiness_checks_the_loaded_model(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        serve.app.state.model = None
        self.assertEqual(self.client.get("/health").status_code, 503)
        self.assertEqual(self.predict().status_code, 503)

    def test_invalid_model_scores_return_500(self):
        for scores in [np.zeros((1, 10)), np.full((1, 11), np.nan), np.full((1, 11), np.inf)]:
            with self.subTest(scores=scores):
                self.model.predict.return_value = scores
                self.assertEqual(self.predict().status_code, 500)


class StartupTests(unittest.TestCase):
    def test_temporary_failure_is_retried(self):
        model = object()
        with patch.object(serve, "load_model", side_effect=[MlflowException("unavailable", INTERNAL_ERROR), model]) as load, patch.object(serve.time, "sleep") as sleep:
            self.assertIs(serve.load_model_with_retry("models:/food11@champion", 2, 0.1), model)
        self.assertEqual(load.call_count, 2)
        sleep.assert_called_once_with(0.1)

    def test_missing_alias_fails_without_retry(self):
        with patch.object(serve, "load_model", side_effect=MlflowException("missing alias", RESOURCE_DOES_NOT_EXIST)) as load, patch.object(serve.time, "sleep") as sleep:
            with self.assertRaises(MlflowException):
                serve.load_model_with_retry("models:/food11@champion", 3, 1)
        self.assertEqual(load.call_count, 1)
        sleep.assert_not_called()

    def test_retries_are_bounded(self):
        with patch.object(serve, "load_model", side_effect=MlflowException("offline", INTERNAL_ERROR)) as load:
            with self.assertRaises(MlflowException):
                serve.load_model_with_retry("models:/food11@champion", 2, 0)
        self.assertEqual(load.call_count, 2)

    def test_lifespan_uses_environment_and_clears_model(self):
        model = SimpleNamespace(metadata=SimpleNamespace(metadata={}))
        with patch.dict("os.environ", {"MODEL_URI": "models:/food11/5", "MLFLOW_TRACKING_URI": "http://mlflow:5000"}), patch.object(serve.mlflow, "set_tracking_uri") as tracking, patch.object(serve, "load_model", return_value=model) as load:
            with TestClient(serve.app):
                self.assertIs(serve.app.state.model, model)
            self.assertIsNone(serve.app.state.model)
            tracking.assert_called_once_with("http://mlflow:5000")
            load.assert_called_once_with("models:/food11/5")

    def test_mismatched_model_labels_reject_startup(self):
        model = SimpleNamespace(metadata=SimpleNamespace(metadata={"class_names": ["wrong"]}))
        with patch.object(serve, "load_model", return_value=model):
            with self.assertRaises(ValueError):
                with TestClient(serve.app):
                    pass

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests


spec = importlib.util.spec_from_file_location("food11_frontend", Path(__file__).resolve().parents[1] / "frontend/app.py")
frontend = importlib.util.module_from_spec(spec)
with patch.dict("sys.modules", {"streamlit": Mock()}):
    spec.loader.exec_module(frontend)


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.upload = SimpleNamespace(name="food.jpg", type="image/jpeg", getvalue=lambda: b"image bytes")

    def test_upload_uses_multipart_and_a_timeout(self):
        response = Mock()
        response.json.return_value = {"category": "Bread", "confidence": 0.8}
        with patch.object(frontend.requests, "post", return_value=response) as post:
            self.assertEqual(frontend.request_prediction(self.upload), response.json.return_value)
        self.assertEqual(post.call_args.kwargs["timeout"], 30)
        self.assertEqual(post.call_args.kwargs["files"]["file"], ("food.jpg", b"image bytes", "image/jpeg"))

    def test_invalid_response_is_rejected(self):
        for payload in [{"category": "Bread", "confidence": 2.3}, {"category": "Bread", "confidence": float("nan")}, []]:
            with self.subTest(payload=payload), patch.object(frontend.requests, "post") as post:
                post.return_value.json.return_value = payload
                with self.assertRaises(ValueError):
                    frontend.request_prediction(self.upload)

    def test_connection_failure_is_shown_on_the_page(self):
        with patch.object(frontend, "st") as st, patch.object(frontend.requests, "post", side_effect=requests.ConnectionError("offline")):
            st.file_uploader.return_value = self.upload
            frontend.main()
            st.error.assert_called_once()
            self.assertIn("unavailable", st.error.call_args.args[0])
            st.write.assert_not_called()

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.food11 import train
from src.food11.data import CATEGORIES


class TrainingTests(unittest.TestCase):
    def test_tracking_uri_comes_from_environment(self):
        with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "http://mlflow:5000"}), patch("sys.argv", ["train"]):
            args = train.parse_args()
        self.assertEqual(args.tracking_uri, "http://mlflow:5000")
        self.assertEqual(args.seed, 42)

    def test_command_line_overrides_tracking_uri(self):
        with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "http://mlflow:5000"}), patch("sys.argv", ["train", "--tracking-uri", "http://localhost:5001", "--seed", "7"]):
            args = train.parse_args()
        self.assertEqual(args.tracking_uri, "http://localhost:5001")
        self.assertEqual(args.seed, 7)

    def test_validation_split_mapping_must_match_training(self):
        good = SimpleNamespace(classes=list(CATEGORIES))
        bad = SimpleNamespace(classes=list(CATEGORIES)[:-1])
        with patch.object(train.datasets, "ImageFolder", side_effect=[good, bad, good]):
            with self.assertRaisesRegex(ValueError, "validation"):
                train.build_dataloaders(Path("unused"), 32)

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.food11 import bootstrap


class BootstrapTests(unittest.TestCase):
    def test_existing_alias_does_not_create_another_version(self):
        existing = SimpleNamespace(version="3")
        with patch.object(bootstrap.mlflow, "MlflowClient") as client, patch.object(bootstrap.mlflow, "start_run") as run:
            client.return_value.get_model_version_by_alias.return_value = existing
            self.assertIs(bootstrap.bootstrap(None, "http://mlflow:5000"), existing)
            run.assert_not_called()

    def test_local_champion_resolves_proxied_artifact_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with closing(sqlite3.connect(root / "mlflow.db")) as db:
                db.execute("CREATE TABLE model_versions (name TEXT, version INTEGER, storage_location TEXT)")
                db.execute("CREATE TABLE registered_model_aliases (name TEXT, version INTEGER, alias TEXT)")
                db.execute("INSERT INTO model_versions VALUES ('food11', 5, 'mlflow-artifacts:/1/models/best/artifacts')")
                db.execute("INSERT INTO registered_model_aliases VALUES ('food11', 5, 'champion')")
                db.commit()
            self.assertEqual(bootstrap.find_local_champion(root), (root / "mlruns/1/models/best/artifacts").resolve())

    def test_missing_local_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                bootstrap.find_local_champion(root)
            self.assertFalse((root / "mlflow.db").exists())

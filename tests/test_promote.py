import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.food11.promote import accuracy_threshold, promote_version


def registry(candidate_accuracy, champion_accuracy=0.8):
    client = Mock()
    client.get_model_version.return_value = SimpleNamespace(version="8", run_id="candidate")
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="5", run_id="champion")
    client.get_run.side_effect = lambda run_id: SimpleNamespace(
        data=SimpleNamespace(metrics={"test_accuracy": candidate_accuracy if run_id == "candidate" else champion_accuracy})
    )
    return client


def test_candidate_at_threshold_is_promoted_and_tagged():
    expected = 0.8 + 2 * math.sqrt(0.8 * 0.2 / 1000)
    client = registry(expected)
    report = promote_version(client, "8", 1000)
    assert report["decision"] == "promoted"
    assert report["threshold"] == pytest.approx(expected)
    client.set_registered_model_alias.assert_called_once_with("food11", "champion", "8")
    client.set_model_version_tag.assert_any_call("food11", "8", "promotion.decision", "promoted")
    client.set_model_version_tag.assert_any_call("food11", "8", "promotion.champion_accuracy", "0.8")


def test_candidate_below_threshold_becomes_challenger():
    client = registry(0.81)
    report = promote_version(client, "8", 1000)
    assert report["decision"] == "challenger"
    client.set_registered_model_alias.assert_called_once_with("food11", "challenger", "8")
    client.set_model_version_tag.assert_any_call("food11", "8", "promotion.candidate_accuracy", "0.81")


def test_missing_metric_leaves_aliases_unchanged():
    client = registry(0.9)
    client.get_run.side_effect = None
    client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={}))
    with pytest.raises(ValueError, match="no test_accuracy"):
        promote_version(client, "8", 1000)
    client.set_registered_model_alias.assert_not_called()


def test_missing_run_id_leaves_aliases_unchanged():
    client = registry(0.9)
    client.get_model_version.return_value.run_id = ""
    with pytest.raises(ValueError, match="no producing run ID"):
        promote_version(client, "8", 1000)
    client.set_registered_model_alias.assert_not_called()


def test_nonfinite_candidate_metric_leaves_aliases_unchanged():
    client = registry(float("nan"))
    with pytest.raises(ValueError, match="invalid test_accuracy"):
        promote_version(client, "8", 1000)
    client.set_registered_model_alias.assert_not_called()


@pytest.mark.parametrize("accuracy,n", [(float("nan"), 100), (1.1, 100), (0.8, 0)])
def test_invalid_gate_inputs_are_rejected(accuracy, n):
    with pytest.raises(ValueError):
        accuracy_threshold(accuracy, n)

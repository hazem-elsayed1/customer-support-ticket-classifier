
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluate import run_evaluation


def test_run_evaluation(tmp_path):
    """Check metrics and JSON output using synthetic predictions."""

    X_test = pd.Series([
        "Example complaint A",
        "Example complaint B",
        "Example complaint C",
        "Example complaint D",
    ])

    y_test = pd.Series(["A", "A", "B", "B"])

    class FakeModel:
        classes_ = np.array(["A", "B"])

        def predict(self, X):
            return np.array(["A", "B", "B", "B"])

    report_path = tmp_path / "test_metrics.json"

    with patch(
        "src.evaluate.train_final_model_with_test_data",
        return_value=(FakeModel(), X_test, y_test),
    ):
        results = run_evaluation(
            "dummy.csv",
            str(report_path),
        )

    assert results["test_samples"] == 4
    assert results["accuracy"] == 0.75
    assert results["macro_f1"] == 0.7333333333333334
    assert results["confusion_matrix"] == [
        [1, 1],
        [0, 2],
    ]

    assert report_path.exists()

    saved_results = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert saved_results == results
    assert "classification_report" in saved_results

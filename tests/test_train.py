
import pandas as pd

from src.train import split_data

from unittest.mock import patch
import numpy as np

from src.train import train_final_model_with_test_data
from src.train import train_model, save_model
from src.predict import load_model

def test_split_data():
    """Verify split sizes, separation, and reproducibility."""

    X = pd.Series(
        [f"Complaint {i}" for i in range(100)]
    )

    y = pd.Series(
        ["A"] * 50 + ["B"] * 50
    )

    first = split_data(X, y)
    second = split_data(X, y)

    X_train, X_val, X_test, y_train, y_val, y_test = first

    # Expected split sizes: 64% train, 16% validation, 20% test
    assert len(X_train) == 64
    assert len(X_val) == 16
    assert len(X_test) == 20

    # All original samples are represented exactly once
    train_ids = set(X_train.index)
    val_ids = set(X_val.index)
    test_ids = set(X_test.index)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    assert train_ids | val_ids | test_ids == set(X.index)

    # Repeated runs produce identical splits
    for first_part, second_part in zip(first, second):
        pd.testing.assert_series_equal(
            first_part,
            second_part,
        )

    # Stratification preserves class proportions
    for labels in (y_train, y_val, y_test):
        proportions = labels.value_counts(normalize=True)
        assert abs(proportions["A"] - 0.5) < 0.01
        assert abs(proportions["B"] - 0.5) < 0.01

def test_model_save_load_roundtrip(tmp_path):
    """Verify that saving and loading preserves predictions."""

    texts = [
        "credit card payment failed",
        "credit card payment declined",
        "credit card billing problem",
        "credit card charged twice",
        "mortgage monthly payment issue",
        "mortgage interest rate problem",
        "mortgage loan payment delayed",
        "mortgage lender complaint",
    ]

    labels = [
        "Credit card",
        "Credit card",
        "Credit card",
        "Credit card",
        "Mortgage",
        "Mortgage",
        "Mortgage",
        "Mortgage",
    ]

    model = train_model(texts, labels)

    sample_complaints = [
        "credit card payment issue",
        "mortgage loan problem",
    ]

    predictions_before = model.predict(sample_complaints)

    model_path = tmp_path / "test_model.joblib"

    save_model(model, str(model_path))

    assert model_path.exists()

    loaded_model = load_model(str(model_path))

    predictions_after = loaded_model.predict(sample_complaints)

    np.testing.assert_array_equal(
        predictions_before,
        predictions_after,
    )



def test_final_training_uses_only_train_validation():
    """Verify final training excludes the held-out test set."""

    X = pd.Series(
        [f"Complaint {i}" for i in range(100)]
    )
    y = pd.Series(["A"] * 50 + ["B"] * 50)

    captured = {}

    def fake_train_model(X_train, y_train):
        captured["train_indices"] = set(X_train.index)
        captured["labels"] = y_train.copy()
        return "fake_model"

    with patch(
        "src.train.load_training_data",
        return_value=(X, y),
    ), patch(
        "src.train.train_model",
        side_effect=fake_train_model,
    ):
        model, X_test, y_test = (
            train_final_model_with_test_data("dummy.csv")
        )

    assert model == "fake_model"
    assert len(captured["train_indices"]) == 80
    assert len(X_test) == 20
    assert len(y_test) == 20

    assert captured["train_indices"].isdisjoint(
        set(X_test.index)
    )

    assert captured["train_indices"] | set(X_test.index) == set(
        X.index
    )

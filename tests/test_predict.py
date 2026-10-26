
import pytest
from unittest.mock import Mock
from src.predict import predict_complaint


def test_predict_rejects_empty_text():
    """Prediction must reject empty complaint text."""

    with pytest.raises(ValueError):
        predict_complaint(None, "   ")


def test_predict_valid_complaint():
    """Prediction should return the model's predicted category."""

    mock_model = Mock()

    mock_model.predict.return_value = ["Credit card"]

    result = predict_complaint(
        mock_model,
        "My credit card payment failed"
    )

    assert result == "Credit card"

    mock_model.predict.assert_called_once_with(
        ["My credit card payment failed"]
    )

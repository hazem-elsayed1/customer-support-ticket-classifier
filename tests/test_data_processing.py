
import pandas as pd

from src.data_processing import remove_missing_narratives
from src.data_processing import (
    remove_conflicting_labels,
    remove_exact_duplicates,
    remove_normalized_duplicates,
    clean_dataset
)

def test_remove_missing_narratives():
    """Remove missing narratives without modifying original data."""

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My payment failed",
            None,
            "My account was blocked"
        ],
        "Product": [
            "Credit card",
            "Mortgage",
            "Checking or savings account"
        ]
    })

    result = remove_missing_narratives(df)

    assert len(result) == 2
    assert result["Consumer complaint narrative"].isna().sum() == 0
    assert len(df) == 3


def test_remove_conflicting_labels():
    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My card was blocked",
            "My card was blocked",
            "My loan was rejected"
        ],
        "Product": [
            "Credit card",
            "Debt collection",
            "Student loan"
        ]
    })

    result = remove_conflicting_labels(df)

    assert len(result) == 1
    assert result["Product"].iloc[0] == "Student loan"


def test_remove_exact_duplicates():
    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My payment failed",
            "My payment failed",
            "My account was blocked"
        ],
        "Product": [
            "Credit card",
            "Credit card",
            "Checking or savings account"
        ]
    })

    result = remove_exact_duplicates(df)

    assert len(result) == 2
    assert result["Consumer complaint narrative"].is_unique
def test_remove_normalized_duplicates():
    """Remove normalized duplicates and preserve original text."""

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My BANK charged me XXXX",
            "  my bank charged me xxxx  ",
            "My loan was rejected"
        ],
        "Product": [
            "Checking or savings account",
            "Checking or savings account",
            "Student loan"
        ]
    })

    result = remove_normalized_duplicates(df)

    assert len(result) == 2
    assert result["normalized_text"].is_unique

    assert result["Consumer complaint narrative"].iloc[0] == (
        "My BANK charged me XXXX"
    )

    assert len(df) == 3


def test_clean_dataset_pipeline():
    """Test all cleaning steps working together."""

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My card was stolen",
            "My card was stolen",
            "My BANK charged me XXXX",
            "my bank charged me xxxx",
            "My loan was rejected",
            None,
            "My account was blocked",
            "My account was blocked"
        ],
        "Product": [
            "Credit card",
            "Credit card",
            "Checking account",
            "Checking account",
            "Student loan",
            "Mortgage",
            "Credit card",
            "Debt collection"
        ]
    })

    result = clean_dataset(df)

    assert len(result) == 3
    assert len(df) == 8
    assert result["normalized_text"].is_unique
    assert result["Consumer complaint narrative"].isna().sum() == 0

    assert set(result["Product"]) == {
        "Credit card",
        "Checking account",
        "Student loan"
    }

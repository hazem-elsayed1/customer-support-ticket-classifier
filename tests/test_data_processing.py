
import pandas as pd
import subprocess
import sys
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


def test_clean_dataset_removes_empty_normalized_text():
    """Texts that become empty after normalization must be removed."""

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "XXXX",
            "   ",
            "My credit card was charged twice.",
        ],
        "Product": [
            "Credit card",
            "Credit card",
            "Credit card",
        ],
    })

    result = clean_dataset(df)

    assert len(result) == 1
    assert (
        result.iloc[0]["Consumer complaint narrative"]
        == "My credit card was charged twice."
    )


def test_clean_dataset_removes_normalized_label_conflicts():
    """Remove all rows with conflicting normalized labels."""

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My card was blocked XXXX",
            "my card was blocked",
            "My mortgage payment is incorrect.",
        ],
        "Product": [
            "Credit card",
            "Checking or savings account",
            "Mortgage",
        ],
    })

    result = clean_dataset(df)

    assert len(result) == 1

    assert (
        result.iloc[0]["Consumer complaint narrative"]
        == "My mortgage payment is incorrect."
    )


def test_data_processing_cli(tmp_path):
    """Verify that the CLI cleans and saves a CSV file."""

    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed" / "cleaned.csv"

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "XXXX",
            "My credit card was charged twice.",
            "My credit card was charged twice.",
            "My mortgage payment was incorrect.",
        ],
        "Product": [
            "Credit card",
            "Credit card",
            "Credit card",
            "Mortgage",
        ],
    })

    df.to_csv(input_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data_processing",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert output_path.is_file()

    cleaned = pd.read_csv(output_path)

    assert len(cleaned) == 2
    assert set(cleaned["Product"]) == {
        "Credit card",
        "Mortgage",
    }

    assert "Original rows: 4" in result.stdout
    assert "Cleaned rows: 2" in result.stdout

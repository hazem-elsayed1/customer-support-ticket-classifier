
import pandas as pd
import re

TEXT_COL = "Consumer complaint narrative"
TARGET_COL = "Product"


def remove_missing_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing complaint narratives."""
    return df.dropna(subset=[TEXT_COL]).copy()


def remove_conflicting_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove narratives associated with multiple product labels."""

    label_counts = df.groupby(TEXT_COL)[TARGET_COL].nunique()

    conflicting_texts = label_counts[label_counts > 1].index

    return df[
        ~df[TEXT_COL].isin(conflicting_texts)
    ].copy()

def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per unique complaint narrative."""

    return df.drop_duplicates(
        subset=[TEXT_COL],
        keep="first"
    ).copy()

def normalize_for_duplicate_check(text: str) -> str:
    """Normalize text for duplicate detection only."""

    text = text.lower()
    text = re.sub(r"\bxxxx\b", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def remove_normalized_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate narratives after normalization."""

    df_clean = df.copy()

    df_clean["normalized_text"] = df_clean[TEXT_COL].apply(
        normalize_for_duplicate_check
    )

    df_clean = df_clean.drop_duplicates(
        subset=["normalized_text"],
        keep="first"
    )

    return df_clean.reset_index(drop=True)

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the complete data cleaning pipeline."""

    df_clean = remove_missing_narratives(df)
    df_clean = remove_conflicting_labels(df_clean)
    df_clean = remove_exact_duplicates(df_clean)

    # Check for conflicting labels after normalization.
    normalized = df_clean[TEXT_COL].apply(
        normalize_for_duplicate_check
    )

    normalized_label_counts = df_clean.groupby(
        normalized
    )[TARGET_COL].nunique()

    if (normalized_label_counts > 1).any():
        raise ValueError(
            "Conflicting labels found after text normalization."
        )

    df_clean = remove_normalized_duplicates(df_clean)

    return df_clean

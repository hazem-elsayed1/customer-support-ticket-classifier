
import pandas as pd
import re

import argparse
from pathlib import Path

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

    # Step 1: Remove missing complaint narratives.
    df_clean = remove_missing_narratives(df)

    # Step 2: Remove texts that become empty after normalization.
    normalized = df_clean[TEXT_COL].apply(
        normalize_for_duplicate_check
    )

    df_clean = df_clean[
        normalized != ""
    ].copy()

    # Step 3: Remove conflicting labels for identical narratives.
    df_clean = remove_conflicting_labels(df_clean)

    # Step 4: Remove exact duplicate narratives.
    df_clean = remove_exact_duplicates(df_clean)


    # Step 5: Remove all normalized conflicting-label groups.
    normalized = df_clean[TEXT_COL].apply(
        normalize_for_duplicate_check
    )

    normalized_label_counts = df_clean.groupby(
        normalized
    )[TARGET_COL].nunique()

    conflicting_normalized_texts = normalized_label_counts[
        normalized_label_counts > 1
    ].index

    df_clean = df_clean[
        ~normalized.isin(conflicting_normalized_texts)
    ].copy()

    # Step 6: Remove remaining normalized duplicates.
    df_clean = remove_normalized_duplicates(df_clean)


    return df_clean



def main():
    """Run the data cleaning pipeline from the command line."""

    parser = argparse.ArgumentParser(
        description="Clean CFPB financial complaint data."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw complaints CSV file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the cleaned CSV file.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    cleaned_df = clean_dataset(df)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_df.to_csv(output_path, index=False)

    print(f"Original rows: {len(df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from src.train import (
    load_training_data,
    split_data,
    train_final_model_with_test_data,
)


def maximum_cosine_similarities(
    test_vectors,
    train_vectors,
    batch_size=100,
):
    """Return the highest training similarity for each test sample."""

    max_similarities = []

    for start in range(0, test_vectors.shape[0], batch_size):
        batch = test_vectors[start:start + batch_size]

        similarities = cosine_similarity(
            batch,
            train_vectors,
            dense_output=False,
        )

        max_values = similarities.max(axis=1).toarray().ravel()
        max_similarities.extend(max_values.tolist())

    return np.asarray(max_similarities)


def analyze_near_duplicates(
    csv_path="data/processed/cleaned_complaints.csv",
    report_path="reports/near_duplicate_analysis.json",
):
    """Measure maximum train-test textual similarity."""

    X, y = load_training_data(csv_path)

    X_train, X_val, X_test, _, _, _ = split_data(X, y)

    # Final training uses both train and validation.
    X_train_val = X.loc[
        X.index.isin(X_train.index.union(X_val.index))
    ]

    vectorizer = TfidfVectorizer(
        min_df=2,
        ngram_range=(1, 2),
    )

    train_vectors = vectorizer.fit_transform(X_train_val)
    test_vectors = vectorizer.transform(X_test)

    scores = maximum_cosine_similarities(
        test_vectors,
        train_vectors,
    )

    # Diagnostic evaluation using the already selected model.
    model, X_test_eval, y_test_eval = (
        train_final_model_with_test_data(csv_path)
    )

    # Verify that the test samples have the same order.
    if not X_test.index.equals(X_test_eval.index):
        raise ValueError("Test sample order does not match.")

    predictions = model.predict(X_test_eval)

    high_similarity_mask = scores >= 0.95
    lower_similarity_mask = ~high_similarity_mask

    def evaluate_group(mask):
        if not mask.any():
            return {
                "samples": 0,
                "accuracy": None,
                "macro_f1": None,
            }

        return {
            "samples": int(mask.sum()),
            "accuracy": float(
                accuracy_score(
                    y_test_eval.iloc[mask],
                    predictions[mask],
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_test_eval.iloc[mask],
                    predictions[mask],
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    group_performance = {
        "high_similarity": evaluate_group(high_similarity_mask),
        "lower_similarity": evaluate_group(lower_similarity_mask),
    }

    group_class_distribution = {
        "high_similarity": {
            str(k): int(v)
            for k, v in y_test_eval.iloc[
                high_similarity_mask
            ].value_counts().items()
        },
        "lower_similarity": {
            str(k): int(v)
            for k, v in y_test_eval.iloc[
                lower_similarity_mask
            ].value_counts().items()
        },
    }

    debt_mask = (
        y_test_eval.to_numpy() == "Debt collection"
    )

    debt_collection_performance = {
        "high_similarity": evaluate_group(
            high_similarity_mask & debt_mask
        ),
        "lower_similarity": evaluate_group(
            lower_similarity_mask & debt_mask
        ),
    }

    report = {
        "train_samples": len(X_train_val),
        "test_samples": len(X_test),
        "similarity_method": "TF-IDF cosine similarity",
        "mean_max_similarity": float(np.mean(scores)),
        "median_max_similarity": float(np.median(scores)),
        "threshold_counts": {
            "0.80": int(np.sum(scores >= 0.80)),
            "0.90": int(np.sum(scores >= 0.90)),
            "0.95": int(np.sum(scores >= 0.95)),
            "0.99": int(np.sum(scores >= 0.99)),
        },
        "group_performance": group_performance,
        "group_class_distribution": group_class_distribution,
        "debt_collection_performance": debt_collection_performance,
    }

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    results = analyze_near_duplicates()
    print(json.dumps(results, indent=2))

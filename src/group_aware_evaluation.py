
import json
from pathlib import Path

import numpy as np
from scipy.sparse import vstack
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit

from src.train import load_training_data, train_model


def create_text_groups(texts, threshold=0.95, batch_size=100):
    """Group near-duplicate texts using batched sparse similarity."""

    texts = list(texts)

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not texts:
        return np.array([], dtype=int)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
    )

    vectors = vectorizer.fit_transform(texts)
    adjacency_batches = []

    for start in range(0, len(texts), batch_size):
        batch = vectors[start:start + batch_size]

        similarities = cosine_similarity(
            batch,
            vectors,
            dense_output=False,
        )

        similarities.data = (
            similarities.data >= threshold
        ).astype(np.int8)

        similarities.eliminate_zeros()
        adjacency_batches.append(similarities)

    adjacency = vstack(
        adjacency_batches,
        format="csr",
    )

    _, group_ids = connected_components(
        adjacency,
        directed=False,
    )

    return group_ids


def group_aware_split(X, y, groups, test_size=0.20, random_state=42):
    """Split data without sharing groups between train and test."""

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y, and groups must have equal lengths.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct groups are required.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    train_groups = set(np.asarray(groups)[train_idx])
    test_groups = set(np.asarray(groups)[test_idx])

    if train_groups & test_groups:
        raise RuntimeError("Group overlap detected.")

    return train_idx, test_idx


def evaluate_group_aware(
    csv_path="data/processed/cleaned_complaints.csv",
    report_path="reports/group_aware_evaluation.json",
    threshold=0.95,
    batch_size=100,
):
    """Evaluate the model on a group-disjoint train/test split."""

    X, y = load_training_data(csv_path)

    groups = create_text_groups(
        X,
        threshold=threshold,
        batch_size=batch_size,
    )

    train_idx, test_idx = group_aware_split(
        X,
        y,
        groups,
        test_size=0.20,
        random_state=42,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    report = {
        "evaluation_type": "near_duplicate_group_aware",
        "similarity_threshold": threshold,
        "total_samples": int(len(X)),
        "total_groups": int(len(np.unique(groups))),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_groups": int(len(np.unique(groups[train_idx]))),
        "test_groups": int(len(np.unique(groups[test_idx]))),
        "group_overlap": int(
            len(
                set(groups[train_idx])
                & set(groups[test_idx])
            )
        ),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "train_class_distribution": {
            str(k): int(v)
            for k, v in y_train.value_counts().items()
        },
        "test_class_distribution": {
            str(k): int(v)
            for k, v in y_test.value_counts().items()
        },
    }

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    examples = [
        "My credit card was charged twice",
        "My credit card was charged twice",
        "My mortgage payment is incorrect",
        "My student loan balance is wrong",
    ]

    groups = create_text_groups(examples)

    for text, group in zip(examples, groups):
        print(f"Group {group}: {text}")

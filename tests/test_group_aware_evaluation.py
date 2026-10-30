
import json

import numpy as np
import pandas as pd
import pytest

from src.group_aware_evaluation import (
    create_text_groups,
    group_aware_split,
    evaluate_group_aware,
)


def test_identical_texts_share_group():
    texts = [
        "My credit card was charged twice",
        "My credit card was charged twice",
        "My mortgage payment is incorrect",
    ]

    groups = create_text_groups(texts)

    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_empty_input():
    groups = create_text_groups([])

    assert isinstance(groups, np.ndarray)
    assert len(groups) == 0


def test_invalid_threshold():
    with pytest.raises(ValueError):
        create_text_groups(
            ["Example complaint"],
            threshold=0,
        )

    with pytest.raises(ValueError):
        create_text_groups(
            ["Example complaint"],
            threshold=1.5,
        )


def test_group_count():
    texts = [
        "My credit card was charged twice",
        "My credit card was charged twice",
        "My mortgage payment is incorrect",
        "My student loan balance is wrong",
    ]

    groups = create_text_groups(texts)

    assert len(np.unique(groups)) == 3


def test_groups_across_different_batches():
    texts = [
        "My credit card was charged twice",
        "My mortgage payment is incorrect",
        "My student loan balance is wrong",
        "My credit card was charged twice",
    ]

    groups = create_text_groups(
        texts,
        threshold=0.95,
        batch_size=2,
    )

    assert groups[0] == groups[3]
    assert groups[0] != groups[1]
    assert groups[1] != groups[2]


def test_batch_size_does_not_change_grouping():
    texts = [
        "My credit card was charged twice",
        "My mortgage payment is incorrect",
        "My credit card was charged twice",
        "My student loan balance is wrong",
        "My mortgage payment is incorrect",
    ]

    groups_small = create_text_groups(
        texts,
        batch_size=1,
    )

    groups_large = create_text_groups(
        texts,
        batch_size=100,
    )

    for i in range(len(texts)):
        for j in range(len(texts)):
            assert (
                groups_small[i] == groups_small[j]
            ) == (
                groups_large[i] == groups_large[j]
            )


def test_group_aware_split_has_no_overlap():
    texts = [
        "Complaint A",
        "Complaint B",
        "Complaint C",
        "Complaint D",
        "Complaint E",
        "Complaint F",
    ]

    labels = [
        "Credit card",
        "Credit card",
        "Mortgage",
        "Debt collection",
        "Debt collection",
        "Student loan",
    ]

    groups = np.array([0, 0, 1, 2, 2, 3])

    train_idx, test_idx = group_aware_split(
        texts,
        labels,
        groups,
        test_size=0.5,
        random_state=42,
    )

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    assert train_groups.isdisjoint(test_groups)

    assert (
        len(train_idx) + len(test_idx)
        == len(texts)
    )

    assert set(train_idx).isdisjoint(
        set(test_idx)
    )


def test_group_aware_evaluation_report(tmp_path):
    texts = [
        f"Complaint number {i} regarding financial account service"
        for i in range(40)
    ]

    labels = [
        "Credit card" if i % 2 == 0 else "Mortgage"
        for i in range(40)
    ]

    df = pd.DataFrame({
        "Consumer complaint narrative": texts,
        "Product": labels,
    })

    csv_path = tmp_path / "complaints.csv"
    report_path = tmp_path / "report.json"

    df.to_csv(csv_path, index=False)

    report = evaluate_group_aware(
        csv_path=str(csv_path),
        report_path=str(report_path),
        threshold=0.95,
        batch_size=10,
    )

    assert report_path.exists()
    assert report["total_samples"] == 40
    assert report["group_overlap"] == 0

    assert (
        report["train_samples"]
        + report["test_samples"]
        == 40
    )

    assert 0 <= report["accuracy"] <= 1
    assert 0 <= report["macro_f1"] <= 1

    saved_report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert saved_report == report

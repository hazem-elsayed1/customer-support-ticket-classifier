
import json

from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

from src.near_duplicate_analysis import maximum_cosine_similarities

def test_near_duplicate_report_consistency():
    """Verify consistency of saved diagnostic results."""

    path = Path("reports/near_duplicate_analysis.json")

    assert path.is_file(), "Near-duplicate report is missing"

    report = json.loads(path.read_text(encoding="utf-8"))

    test_samples = report["test_samples"]
    counts = report["threshold_counts"]

    assert (
        0 <= counts["0.99"]
        <= counts["0.95"]
        <= counts["0.90"]
        <= counts["0.80"]
        <= test_samples
    )

    groups = report["group_performance"]

    assert (
        groups["high_similarity"]["samples"]
        + groups["lower_similarity"]["samples"]
        == test_samples
    )

    distributions = report["group_class_distribution"]

    assert (
        sum(distributions["high_similarity"].values())
        == groups["high_similarity"]["samples"]
    )

    assert (
        sum(distributions["lower_similarity"].values())
        == groups["lower_similarity"]["samples"]
    )

def test_maximum_cosine_similarities():
    """Check maximum cosine similarity for synthetic vectors."""

    train_vectors = csr_matrix([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    test_vectors = csr_matrix([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])

    scores = maximum_cosine_similarities(
        test_vectors,
        train_vectors,
        batch_size=2,
    )

    expected = np.array([
        1.0,
        1.0,
        1 / np.sqrt(2),
        0.0,
    ])

    np.testing.assert_allclose(
        scores,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )

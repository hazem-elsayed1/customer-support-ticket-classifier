
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.train import train_final_model_with_test_data


def run_evaluation(
    csv_path: str,
    report_path: str = "reports/test_metrics.json",
) -> dict:
    """Train the selected model and evaluate on held-out test data."""

    model, X_test, y_test = train_final_model_with_test_data(
        csv_path
    )

    predictions = model.predict(X_test)
    labels = [str(label) for label in model.classes_]

    results = {
        "test_samples": len(y_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        ),
        "labels": labels,
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        ).tolist(),
    }

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return results


if __name__ == "__main__":
    results = run_evaluation(
        "data/processed/cleaned_complaints.csv"
    )

    print(f"Test samples: {results['test_samples']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print("Report saved to reports/test_metrics.json")

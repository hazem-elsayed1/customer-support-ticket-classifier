
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline


def load_model(model_path: str) -> Pipeline:
    """Load a previously trained classification pipeline."""

    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    model = joblib.load(path)

    return model

def predict_complaint(model: Pipeline, text: str) -> str:
    """Predict the product category for a complaint."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Complaint text must be a non-empty string.")

    prediction = model.predict([text])
    return str(prediction[0])

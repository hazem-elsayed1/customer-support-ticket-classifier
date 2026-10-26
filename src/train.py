
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import joblib

def build_model() -> Pipeline:
    """Create the selected TF-IDF + Logistic Regression pipeline."""

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            min_df=2,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=2.0
        ))
    ])

    return model

def train_model(X_train, y_train) -> Pipeline:
    """Build and train the classifier."""

    model = build_model()

    model.fit(X_train, y_train)

    return model

def load_training_data(csv_path: str):
    """Load cleaned complaint texts and labels."""

    df = pd.read_csv(csv_path)

    X = df["Consumer complaint narrative"]
    y = df["Product"]

    return X, y

def split_data(X, y):
    """Create reproducible train, validation, and test splits."""

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        stratify=y_train_val,
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def run_training(csv_path: str):
    """Load data, split it, and train the classifier."""

    X, y = load_training_data(csv_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y
    )

    model = train_model(X_train, y_train)

    return model, X_val, y_val, X_test, y_test

def evaluate_model(model, X_eval, y_eval) -> dict:
    """Evaluate a trained classifier."""

    predictions = model.predict(X_eval)

    accuracy = accuracy_score(y_eval, predictions)

    macro_f1 = f1_score(
        y_eval,
        predictions,
        average="macro"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

def save_model(model: Pipeline, model_path: str) -> None:
    """Save a trained pipeline to disk."""

    path = Path(model_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, path)

def train_final_model(csv_path: str) -> Pipeline:
    """Train the selected model on train + validation data."""

    X, y = load_training_data(csv_path)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    model = train_model(X_train_val, y_train_val)

    return model

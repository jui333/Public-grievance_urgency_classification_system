from pathlib import Path
from typing import Tuple
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer


def build_text_classifier(df: pd.DataFrame, text_column: str, target_column: str) -> Tuple[Pipeline, pd.DataFrame]:
    df = df[[text_column, target_column]].dropna().copy()
    df = df[df[target_column].astype(str).str.len() > 2]
    df = df[df[target_column].astype(str) != "Unknown"]
    if len(df) > 12000:
        df = df.sample(12000, random_state=42)
    elif len(df) < 200:
        raise ValueError("Not enough labeled data for training.")

    X = df[text_column].astype(str)
    y = df[target_column].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    results = pd.DataFrame({"text": X_test, "actual": y_test, "predicted": predictions})
    results.reset_index(drop=True, inplace=True)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Weighted F1: {f1:.3f}")
    return pipeline, results


def build_category_model(df: pd.DataFrame, text_column: str = "complaint_text_clean", target_column: str = "category_label") -> Tuple[Pipeline, pd.DataFrame]:
    df = df[df[target_column].astype(str) != "Unknown"].copy()
    df = df[df[target_column].astype(str).str.len() > 2].copy()
    if df.empty:
        raise ValueError("No category labels available for training.")
    return build_text_classifier(df, text_column, target_column)


def build_urgency_model(df: pd.DataFrame, text_column: str = "complaint_text_clean", target_column: str = "urgency_label") -> Tuple[Pipeline, pd.DataFrame]:
    return build_text_classifier(df, text_column, target_column)


def save_model(model: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(input_path: Path) -> Pipeline:
    return joblib.load(input_path)


def predict_category(model: Pipeline, text: str) -> str:
    return model.predict([text])[0]

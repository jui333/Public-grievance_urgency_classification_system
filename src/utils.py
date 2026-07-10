import re
from pathlib import Path
from typing import Tuple, Dict, Optional
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.preprocessing import (
    create_text_feature,
    create_target_label,
    map_urgency_label,
    clean_text,
)

ROOT = Path(__file__).resolve().parents[1]
CATEGORY_MODEL_PATH = ROOT / "src" / "category_model.joblib"
URGENCY_MODEL_PATH = ROOT / "src" / "urgency_model.joblib"
DATA_PATH = ROOT / "data" / "sample_or_raw_data.csv"

# ----------------- From backend/model.py -----------------

def build_text_classifier(df: pd.DataFrame, text_column: str, target_column: str) -> Tuple[Pipeline, pd.DataFrame]:
    print(f"\nProcessing data for {target_column}...")
    df = df[[text_column, target_column]].dropna().copy()
    df = df[df[target_column].astype(str).str.len() > 2]
    df = df[df[target_column].astype(str) != "Unknown"]
    if len(df) > 25000:
        df = df.sample(25000, random_state=42)
    elif len(df) < 200:
        raise ValueError("Not enough labeled data for training.")
        
    # Keep only classes with at least 10 examples to support CV splits in train_test_split
    counts = df[target_column].value_counts()
    df = df[df[target_column].isin(counts[counts >= 10].index)]
    
    print(f"Data loaded: {len(df)} samples")

    X = df[text_column].astype(str)
    y = df[target_column].astype(str)
    
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    print("Building and training model...")
    # Log class distribution for insight
    from collections import Counter
    class_counts = Counter(y_train)
    print("Training class distribution:")
    for cls, cnt in class_counts.items():
        cls_safe = str(cls).encode('ascii', 'ignore').decode('ascii')
        print(f"   - {cls_safe}: {cnt}")

    # Build richer feature set: word n‑grams + character n‑grams
    from sklearn.pipeline import FeatureUnion
    word_vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        analyzer="word",
    )
    char_vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(3, 5),
        sublinear_tf=True,
        analyzer="char_wb",
    )
    combined_features = FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])

    pipeline = Pipeline(
        [
            ("features", combined_features),
            ("clf", CalibratedClassifierCV(
                LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1),
                cv=3,
            )),
        ]
    )
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model on test set...")
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    results = pd.DataFrame({"text": X_test, "actual": y_test, "predicted": predictions})
    results.reset_index(drop=True, inplace=True)
    print(f"\n{target_column.replace('_', ' ').title()}:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Weighted F1: {f1:.3f}\n")
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


# ----------------- From backend/predict.py -----------------

EMERGENCY_KEYWORDS = [
    "ambulance",
    "emergency",
    "accident",
    "patient",
    "medical",
    "hospital",
    "life",
    "urgent",
    "urgent help",
    "need help",
    "critical",
]

EMERGENCY_CATEGORY_OVERRIDES = {
    "ambulance": "Public Health & Safety",
    "emergency": "Public Health & Safety",
}

CATEGORY_TO_ACTION = {
    "Unknown": "Review the complaint and route to the appropriate department.",
    "Billing": "Escalate to billing services and follow up quickly.",
    "Public Safety": "Alert the safety department and prioritize response.",
    "Health": "Forward to health services with high priority.",
    "Emergency Services": "Dispatch emergency responders immediately.",
    "Public Works / Civil Infrastructure": "Route this to the public works or civil infrastructure team for inspection.",
    "Sanitation & Solid Waste Management": "Route this to the sanitation and solid waste team for cleanup or enforcement.",
    "Water Supply & Sewerage": "Route this to the water and sewerage department for urgent repair.",
    "Electricity / Power Department": "Route this to the electricity or power department for immediate safety action.",
    "Public Health & Safety": "Route this to the public health and safety department for follow-up.",
}


def detect_emergency(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in EMERGENCY_KEYWORDS)


def resolve_emergency_category(text: str, predicted_category: str) -> str:
    text_lower = text.lower()
    for keyword, override_category in EMERGENCY_CATEGORY_OVERRIDES.items():
        if keyword in text_lower:
            return override_category
    return predicted_category


def format_recommendation(category: str, urgency: str, confidence_str: str, override: bool) -> Dict[str, str]:
    action = CATEGORY_TO_ACTION.get(
        category,
        "Review this complaint and assign it to the correct grievance officer.",
    )
    
    timeline = ""
    if urgency == "High":
        timeline = "URGENT (Resolve within 24 hours) - "
    elif urgency == "Medium":
        timeline = "MEDIUM (Resolve within 3 days) - "
    elif urgency == "Low":
        timeline = "LOW (Resolve within 7 days) - "
        
    action = timeline + action
    explanation = f"This complaint is classified as {category} with {urgency} urgency."
    if override:
        explanation += " Escalated automatically due to emergency keywords."
        
    return {
        "category": category,
        "urgency": urgency,
        "action": action,
        "explanation": explanation,
        "confidence": confidence_str
    }


def analyze_complaint(text: str, category_model, urgency_model) -> Dict[str, str]:
    if len(text.strip()) < 10:
        return {
            "category": "Unknown",
            "urgency": "Unknown",
            "action": "Please provide a more detailed complaint for accurate classification.",
            "explanation": "The text provided is too short to analyze.",
            "confidence": "N/A"
        }

    cleaned = clean_text(text)
    
    cat_pred = category_model.predict([cleaned])[0]
    cat_proba = category_model.predict_proba([cleaned])[0].max()
    
    # Resolve numeric category prediction to description if applicable
    if str(cat_pred).replace(".", "", 1).isdigit():
        from src.preprocessing import load_category_mapping
        mapping = load_category_mapping()
        cat_desc = mapping.get(str(int(float(cat_pred))), str(cat_pred))
    else:
        cat_desc = str(cat_pred)
    
    urg_pred = urgency_model.predict([cleaned])[0]
    urg_proba = urgency_model.predict_proba([cleaned])[0].max()
    
    override = False
    if detect_emergency(text):
        cat_desc = resolve_emergency_category(text, cat_desc)
        urg_pred = "High"
        urg_proba = 1.0
        override = True
        
    avg_confidence = (cat_proba + urg_proba) / 2
    confidence_str = f"{avg_confidence * 100:.1f}%"
    
    return format_recommendation(cat_desc, urg_pred, confidence_str, override)


# ----------------- From backend/model_service.py -----------------

def prepare_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = create_text_feature(df)
    df = create_target_label(df)
    df = map_urgency_label(df)
    return df


def train_models() -> tuple:
    # Use tqdm to display progress of training steps
    with tqdm(total=4, desc='Training Models', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]') as pbar:
        df = prepare_training_data()
        pbar.update(1)
        category_model, category_results = build_category_model(df)
        pbar.update(1)
        urgency_model, urgency_results = build_urgency_model(df)
        pbar.update(1)
        save_model(category_model, CATEGORY_MODEL_PATH)
        save_model(urgency_model, URGENCY_MODEL_PATH)
        pbar.update(1)
    return category_model, urgency_model, category_results, urgency_results


def train_model() -> tuple:
    category_model, _, category_results, _ = train_models()
    return category_model, category_results


def get_category_model():
    if CATEGORY_MODEL_PATH.exists():
        return load_model(CATEGORY_MODEL_PATH)
    return train_models()[0]


def get_urgency_model():
    if URGENCY_MODEL_PATH.exists():
        return load_model(URGENCY_MODEL_PATH)
    return train_models()[1]


def predict_complaint(text: str) -> dict:
    category_model = get_category_model()
    urgency_model = get_urgency_model()
    return analyze_complaint(text, category_model, urgency_model)

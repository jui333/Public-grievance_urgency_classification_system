from pathlib import Path
import pandas as pd
from backend.preprocessing import (
    create_text_feature,
    create_target_label,
    map_urgency_label,
    clean_text,
)
from backend.model import build_category_model, build_urgency_model, save_model, load_model
from backend.predict import analyze_complaint

ROOT = Path(__file__).resolve().parents[1]
CATEGORY_MODEL_PATH = ROOT / "backend" / "category_model.joblib"
URGENCY_MODEL_PATH = ROOT / "backend" / "urgency_model.joblib"
DATA_PATH = ROOT / "no_pii_grievance.json"


def prepare_training_data() -> pd.DataFrame:
    df = pd.read_json(DATA_PATH)
    df = create_text_feature(df)
    df = create_target_label(df)
    df = map_urgency_label(df)
    return df


def train_models() -> tuple:
    df = prepare_training_data()
    category_model, category_results = build_category_model(df)
    urgency_model, urgency_results = build_urgency_model(df)
    save_model(category_model, CATEGORY_MODEL_PATH)
    save_model(urgency_model, URGENCY_MODEL_PATH)
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
    return analyze_complaint(text, category_model, urgency_model.predict([clean_text(text)])[0])

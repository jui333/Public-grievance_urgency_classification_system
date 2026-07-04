import re
from pathlib import Path
from typing import Optional
import pandas as pd


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_grievance_data(path: str) -> pd.DataFrame:
    df = pd.read_json(path)
    return df


def load_category_mapping(mapping_path: Optional[str] = None) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    mapping_path = Path(mapping_path or project_root / "CategoryCode_Mapping.xlsx")
    if not mapping_path.exists():
        return {}

    try:
        mapping_df = pd.read_excel(mapping_path, sheet_name="Complaint Category")
    except Exception:
        return {}

    mapping = {}
    for _, row in mapping_df.dropna(subset=["Code", "Description"]).iterrows():
        code = row["Code"]
        description = str(row["Description"]).strip()
        if pd.notna(code) and description:
            mapping[str(int(float(code)))] = description
    return mapping


def create_text_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    subject = df.get("subject_content_text", pd.Series([""] * len(df)))
    remarks = df.get("remarks_text", pd.Series([""] * len(df)))
    subject = subject.fillna("").astype(str)
    remarks = remarks.fillna("").astype(str)
    df["complaint_text"] = subject + " " + remarks
    df["complaint_text_clean"] = df["complaint_text"].apply(clean_text)
    return df


def create_target_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = load_category_mapping()

    CATEGORY_TEXT_OVERRIDES = {
        "nil availability": "Water Supply & Sewerage",
        "no availability": "Water Supply & Sewerage",
        "water shortage": "Water Supply & Sewerage",
        "lack of water": "Water Supply & Sewerage",
        "water issue": "Water Supply & Sewerage",
        "water cut": "Water Supply & Sewerage",
        "water supply issue": "Water Supply & Sewerage",
    }

    def normalize_category(value):
        if pd.isna(value):
            return "Unknown"
        text_value = str(value).strip()
        if not text_value:
            return "Unknown"
        normalized_text = re.sub(r"\s+", " ", text_value.lower()).strip()
        
        if normalized_text in CATEGORY_TEXT_OVERRIDES:
            return CATEGORY_TEXT_OVERRIDES[normalized_text]
            
        if normalized_text.replace(".", "", 1).isdigit():
            code = str(int(float(normalized_text)))
            desc = mapping.get(code, "Unknown")
            desc_norm = re.sub(r"\s+", " ", desc.lower()).strip()
            if desc_norm in CATEGORY_TEXT_OVERRIDES:
                return CATEGORY_TEXT_OVERRIDES[desc_norm]
            return desc
            
        return text_value

    if "CategoryV7" in df.columns:
        df["category_label"] = df["CategoryV7"].apply(normalize_category)
    else:
        df["category_label"] = "Unknown"

    df["category_label"] = df["category_label"].fillna("Unknown").astype(str)
    return df


def map_urgency_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def urgency_rules(text: str, category) -> str:
        text = str(text or "")
        category = str(category or "")
        text_lower = text.lower()

        water_critical_keywords = [
            "water shortage",
            "no water",
            "water cut",
            "dry tap",
            "tap dry",
            "nil availability",
            "no availability",
            "water outage",
            "no drinking water",
            "water supply",
        ]
        high_keywords = ["fire", "fraud", "accident", "danger", "urgent", "security", "illegal", "death", "cut off"]
        medium_keywords = ["delay", "service", "request", "support", "help", "issue"]

        if any(word in text_lower for word in water_critical_keywords):
            return "High"
        if any(word in text_lower for word in high_keywords):
            return "High"
        if any(word in text_lower for word in medium_keywords):
            return "Medium"
        if category.lower() in ["public safety", "health"]:
            return "High"
        return "Low"

    df["urgency_label"] = df.apply(
        lambda row: urgency_rules(row["complaint_text"], row.get("CategoryV7", "")),
        axis=1,
    )
    return df

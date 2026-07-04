from typing import Dict
from backend.preprocessing import clean_text


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
        timeline = "🚨 URGENT (Resolve within 24 hours) - "
    elif urgency == "Medium":
        timeline = "⏳ MEDIUM (Resolve within 3 days) - "
    elif urgency == "Low":
        timeline = "✅ LOW (Resolve within 7 days) - "
        
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
    
    urg_pred = urgency_model.predict([cleaned])[0]
    urg_proba = urgency_model.predict_proba([cleaned])[0].max()
    
    override = False
    if detect_emergency(text):
        cat_pred = resolve_emergency_category(text, cat_pred)
        urg_pred = "High"
        urg_proba = 1.0
        override = True
        
    avg_confidence = (cat_proba + urg_proba) / 2
    confidence_str = f"{avg_confidence * 100:.1f}%"
    
    return format_recommendation(cat_pred, urg_pred, confidence_str, override)

import sys
import pathlib
# Ensure the project root (parent of src) is on PYTHONPATH for absolute imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from fastapi import FastAPI
from pydantic import BaseModel
from src.utils import predict_complaint, train_models, CATEGORY_MODEL_PATH

app = FastAPI(title="Public Grievance Urgency Microservice")

class ComplaintRequest(BaseModel):
    text: str

class ComplaintResponse(BaseModel):
    category: str
    urgency: str
    action: str
    explanation: str
    confidence: str

@app.get("/")
def root():
    return {"status": "ok", "service": "Public Grievance Urgency Microservice"}

@app.post("/predict", response_model=ComplaintResponse)
def predict(request: ComplaintRequest):
    return predict_complaint(request.text)

def run_training():
    print("\n" + "="*60)
    print("🚀 Starting Model Training...")
    print("="*60 + "\n")
    
    _, _, category_results, _ = train_models()
    
    print("\n" + "="*60)
    print(f"✅ Saved trained models to {CATEGORY_MODEL_PATH.parent}")
    print("="*60 + "\n")
    print("Sample predictions:")
    print(category_results.head())

if __name__ == "__main__":
    run_training()

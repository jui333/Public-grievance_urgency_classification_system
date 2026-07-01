from fastapi import FastAPI
from pydantic import BaseModel
from backend.model_service import predict_complaint

app = FastAPI(title="Public Grievance Urgency Microservice")

class ComplaintRequest(BaseModel):
    text: str

class ComplaintResponse(BaseModel):
    category: str
    urgency: str
    action: str
    explanation: str

@app.get("/")
def root():
    return {"status": "ok", "service": "Public Grievance Urgency Microservice"}

@app.post("/predict", response_model=ComplaintResponse)
def predict(request: ComplaintRequest):
    return predict_complaint(request.text)

# Public Grievance Urgency Classification System

## Project Overview
This project builds a complaint classification and urgency recommendation system for public grievance data. It uses text-based grievance records to predict complaint categories, score urgency, and recommend escalation actions.

## Repository Structure
- `data/` - raw and processed dataset files
- `notebooks/` - exploratory analysis and model development
- `backend/` - reusable preprocessing, modeling, and API logic
- `frontend/` - user-facing demo interface
- `docs/` - project report and documentation

## How to run
1. Create and activate the virtual environment:
   - `python3 -m venv public_grievance_urgency_classification_system_venv`
   - `source public_grievance_urgency_classification_system_venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model once: `python backend/train.py`
4. Start the backend microservice:
   - `uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000`
5. Start the frontend demo:
   - `streamlit run frontend/app.py`

## Notes
## Dataset availability
- The dataset files are not included in this repository.
- Place the following files in the repository root before running the project:
  - `CategoryCode_Mapping.xlsx`
  - `no_pii_action_history.json`
  - `no_pii_grievance.json`
- Download the dataset from the provider link or shared source for this project.
- The frontend uses the backend API at `http://127.0.0.1:8000/predict`.
- The backend persists the trained model at `backend/category_model.joblib`.

## Notes
## Dataset availability
- The dataset files are not included in this repository.
- Place the following files in the repository root before running the project:
  - `CategoryCode_Mapping.xlsx`
  - `no_pii_action_history.json`
  - `no_pii_grievance.json`
- Download the dataset from the provider link or shared source for this project.
- The dataset files are already present in the repository root.
- Notebooks are used for model exploration, while backend code is used for reusable logic.

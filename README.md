# Public Grievance Urgency Classification System

## Project Overview
This project builds a complaint classification and urgency recommendation system for public grievance data. It uses text-based grievance records to predict complaint categories, score urgency, and recommend escalation actions.

## Repository Structure
- `data/` - raw and processed dataset files (such as `sample_or_raw_data.csv`)
- `notebooks/` - exploratory analysis and model development (`exploration_or_modeling.ipynb`)
- `src/` - reusable preprocessing, modeling, and API microservice logic (`main.py`, `preprocessing.py`, `utils.py`)
- `app/` - user-facing Streamlit application (`app.py`)
- `docs/` - project report and presentation documentation

## How to run
1. Create and activate the virtual environment:
   - `python3 -m venv public_grievance_urgency_classification_system_venv`
   - `source public_grievance_urgency_classification_system_venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model once:
   - `PYTHONPATH=. python src/main.py`
4. Start the backend microservice:
   - `uvicorn src.main:app --reload --host 127.0.0.1 --port 8000`
5. Start the frontend demo:
   - `streamlit run app/app.py`

Alternatively, you can start both the backend API and frontend UI using the startup script:
- `./run.sh`

## Notes
- The frontend uses the backend API at `http://127.0.0.1:8000/predict`.
- The backend persists the trained models in `src/category_model.joblib` and `src/urgency_model.joblib`.

## Dataset availability
- Place the following file in the `data/` directory before running the project:
  - `sample_or_raw_data.csv`
- The dataset file is not included in this repository. Download the dataset from the provider link or shared source for this project.

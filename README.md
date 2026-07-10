# Public Grievance Urgency Classification System

## Project Overview
This project builds an automated complaint classification, urgency rating, and escalation recommendation system for public grievance text data. It predicts grievance categories, scores urgency, and recommends SLAs/escalation actions using a calibrated TF-IDF and Logistic Regression model pipeline.

## Repository Structure
The project is structured according to the following layout:
```
public_grievance_urgency_classification_system/
│
├── data/
│   ├── sample_or_raw_data.csv          # Structured dataset file
│   ├── no_pii_grievance.json           # Raw JSON grievance data
│   └── no_pii_action_history.json      # Raw JSON action history data
│
├── notebooks/
│   └── exploration_or_modeling.ipynb  # Exploratory analysis and model exploration notebook
│
├── src/
│   ├── main.py                         # FastAPI web API microservice & model training runner
│   ├── preprocessing.py                # Text normalization and target mapping logic
│   └── utils.py                        # Model training, inference, and classification logic
│
├── app/
│   └── app.py                          # User-facing Streamlit application dashboard
│
├── docs/
│   ├── project_report.md               # Summary report detailing system approach
│   └── presentation.pdf                # Compilation of PDF presentation slides
│
├── requirements.txt                    # Project package dependencies
└── README.md                           # Main documentation file
```

## Setup and Installation

### 1. Create and Activate Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv public_grievance_urgency_classification_system_venv
  .\public_grievance_urgency_classification_system_venv\Scripts\Activate.ps1
  ```
- **Linux/macOS:**
  ```bash
  python3 -m venv public_grievance_urgency_classification_system_venv
  source public_grievance_urgency_classification_system_venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset CSV
Ensure the raw JSON data file `data/no_pii_grievance.json` is present, then generate the CSV dataset:
- **Windows:**
  ```powershell
  .\public_grievance_urgency_classification_system_venv\Scripts\python.exe scratch/convert_dataset.py
  ```
- **Linux/macOS:**
  ```bash
  python3 scratch/convert_dataset.py
  ```

## How to Run

### 1. Train the Models
To train the classifiers on the newly generated dataset:
- **Windows (PowerShell):**
  ```powershell
  $env:PYTHONPATH="."
  .\public_grievance_urgency_classification_system_venv\Scripts\python.exe src/main.py
  ```
- **Linux/macOS:**
  ```bash
  PYTHONPATH=. python3 src/main.py
  ```

### 2. Run the Backend API Service
Start the FastAPI service locally on port 8000:
- **Windows:**
  ```powershell
  .\public_grievance_urgency_classification_system_venv\Scripts\uvicorn src.main:app --port 8000
  ```
- **Linux/macOS:**
  ```bash
  uvicorn src.main:app --port 8000
  ```

### 3. Run the Frontend Dashboard App
Start the Streamlit dashboard on port 8501:
- **Windows:**
  ```powershell
  $env:STREAMLIT_SERVER_HEADLESS="true"
  .\public_grievance_urgency_classification_system_venv\Scripts\streamlit run app/app.py
  ```
- **Linux/macOS:**
  ```bash
  STREAMLIT_SERVER_HEADLESS=true streamlit run app/app.py
  ```

## Notes
- The Streamlit frontend calls the backend API at `http://127.0.0.1:8000/predict`.
- The trained classifiers are persisted as joblib serialization binaries at `src/category_model.joblib` and `src/urgency_model.joblib`.


# Project Report

## Public Grievance Urgency Classification System

### Problem Statement
The system classifies grievance complaints into categories and assigns urgency labels to help government and civic agencies prioritize responses.

### Dataset
The project uses grievance records from the provided `no_pii_grievance.json` dataset. Complaint text is derived from `subject_content_text` and `remarks_text`.

### Approach
- Cleaned and combined complaint text fields
- Created target category labels from `CategoryV7`
- Built a rule-based urgency label mapper
- Trained a text classification model using TF-IDF and logistic regression
- Built a Streamlit frontend for sample prediction and recommendations

### Limitations
- Urgency is derived from simple keyword rules and may not capture case nuance
- The model is trained on a single dataset and may need more data for better generalization
- The app uses a sample data load path and is not yet optimized for production deployment

### Future Improvements
- Add a dedicated API backend using FastAPI
- Use a more robust urgency model or a multi-task classifier
- Add department-specific escalation mapping from category codes

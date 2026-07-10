import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/"


def is_backend_ready() -> bool:
    try:
        response = requests.get(HEALTH_URL, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def main():
    st.set_page_config(page_title="Grievance Classifier", page_icon="⚖️", layout="centered")
    st.title("Public Grievance Urgency Classification")
    st.caption("Enter a civic complaint and receive a predicted category, urgency level, and action recommendation.")

    backend_ready = is_backend_ready()
    if not backend_ready:
        st.warning(
            "Backend API is not reachable at http://127.0.0.1:8000. Start the backend first or run `./run.sh` from the project root."
        )

    complaint_input = st.text_area(
        "Complaint text",
        "The water supply has been cut off for three days and no response from the municipality.",
    )

    if st.button("Analyze Complaint", use_container_width=True, disabled=not backend_ready):
        try:
            response = requests.post(API_URL, json={"text": complaint_input})
            response.raise_for_status()
            result = response.json()

            st.markdown("### Prediction")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Category", result["category"])
            with col2:
                st.metric("Urgency", result["urgency"])
            with col3:
                st.metric("Confidence", result.get("confidence", "N/A"))

            st.success(result["action"])
            st.info(result["explanation"])
        except requests.exceptions.RequestException as exc:
            st.error(f"Failed to call backend API: {exc}")


if __name__ == "__main__":
    main()

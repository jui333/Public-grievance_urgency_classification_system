import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"


def main():
    st.set_page_config(page_title="Grievance Classifier", page_icon="⚖️", layout="centered")
    st.title("Public Grievance Urgency Classification")
    st.caption("Enter a civic complaint and receive a predicted category, urgency level, and action recommendation.")

    complaint_input = st.text_area(
        "Complaint text",
        "The water supply has been cut off for three days and no response from the municipality.",
    )

    if st.button("Analyze Complaint", use_container_width=True):
        try:
            response = requests.post(API_URL, json={"text": complaint_input})
            response.raise_for_status()
            result = response.json()

            st.markdown("### Prediction")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Category", result["category"])
            with col2:
                st.metric("Urgency", result["urgency"])

            st.success(result["action"])
            st.info(result["explanation"])
        except requests.exceptions.RequestException as exc:
            st.error(f"Failed to call backend API: {exc}")


if __name__ == "__main__":
    main()

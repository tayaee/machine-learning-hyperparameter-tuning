import streamlit as st
from predict import VisaPredictor
import os

st.set_page_config(page_title="Visa Prediction", layout="wide")
st.title("Visa Approval Predictor")

model_dir = "data"
if os.path.exists(model_dir):
    model_files = [
        f.replace(".joblib", "") for f in os.listdir(model_dir) if f.endswith(".joblib") and f != "model_columns"
    ]
    selected_model = st.sidebar.selectbox("Select Model", sorted(model_files))
else:
    st.error("Model directory not found. Please run train.py first.")
    st.stop()

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        continent = st.selectbox("Continent", ["Asia", "Africa", "Europe", "North America", "South America", "Oceania"])
        education = st.selectbox("Education", ["Bachelor's", "Master's", "Doctorate", "High School"])
        job_experience = st.selectbox("Job Experience", ["Y", "N"])
        training = st.selectbox("Job Training", ["Y", "N"])
    with col2:
        no_of_employees = st.number_input("Employees", min_value=1)
        prevailing_wage = st.number_input("Wage")
        unit_of_wage = st.selectbox("Unit", ["Year", "Month", "Week", "Hour"])
        region_of_employment = st.selectbox("Region", ["West", "Northeast", "South", "Midwest", "Island"])
    submit = st.form_submit_button("Predict")

if submit:
    try:
        predictor = VisaPredictor(selected_model)
        user_input = {
            "continent": continent,
            "education_of_employee": education,
            "has_job_experience": job_experience,
            "requires_job_training": training,
            "no_of_employees": no_of_employees,
            "prevailing_wage": prevailing_wage,
            "unit_of_wage": unit_of_wage,
            "region_of_employment": region_of_employment,
        }
        result, prob = predictor.predict(user_input)
        if result == "Certified":
            st.success(f"Result: {result} ({prob:.2%})")
        else:
            st.error(f"Result: {result} ({prob:.2%})")
    except Exception as e:
        st.error(f"Error: {e}")

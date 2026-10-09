import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Chronic Disease Multi-Morbidity Risk", layout="wide")

@st.cache_resource
def load_pipeline():
    bundle = joblib.load('chronic_disease_model.joblib')
    return bundle['preprocessor'], bundle['model'], bundle['classes']

preprocessor, model, classes = load_pipeline()

st.title("🏥 Multi-Class Chronic Disease Risk Engine")
st.markdown("Clinical triage risk assessment across 8 single & comorbid disease states.")

# Sidebar inputs
st.sidebar.header("Patient Vitals & Demographics")
age = st.sidebar.slider("Age", 18, 90, 45)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 26.5)
glucose = st.sidebar.number_input("Fasting Glucose (mg/dL)", 60.0, 300.0, 105.0)
hba1c = st.sidebar.slider("HbA1c Level (%)", 4.0, 14.0, 5.7)
cholesterol = st.sidebar.number_input("Total Cholesterol", 100.0, 400.0, 195.0)
triglycerides = st.sidebar.number_input("Triglycerides", 50.0, 500.0, 150.0)

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
smoking = st.sidebar.selectbox("Smoking Status", ["Never", "Former", "Current"])
activity = st.sidebar.selectbox("Physical Activity", ["Low", "Moderate", "High"])
stress = st.sidebar.selectbox("Stress Level", ["Low", "Moderate", "High"])
education = st.sidebar.selectbox("Education Level", ["High School", "Graduate", "Postgraduate"])
employment = st.sidebar.selectbox("Employment Status", ["Employed", "Unemployed", "Retired"])

# Initialize full input schema expected by preprocessor
expected_columns = preprocessor.feature_names_in_
input_data = pd.DataFrame({col: [np.nan] for col in expected_columns})

user_inputs = {
    'age': age,
    'bmi': bmi,
    'glucose': glucose,
    'HbA1c_level': hba1c,
    'cholesterol': cholesterol,
    'triglycerides': triglycerides,
    'gender': gender,
    'smoking': smoking,
    'physical_activity': activity,
    'stress_level': stress,
    'education_level': education,
    'employment_status': employment
}

for col, val in user_inputs.items():
    if col in input_data.columns:
        input_data[col] = val

if st.button("Assess Comorbidity Risk"):
    # Preprocess & Predict
    proc_input = preprocessor.transform(input_data)
    pred_idx = model.predict(proc_input)[0]
    probs = model.predict_proba(proc_input)[0]
    
    pred_label = classes[pred_idx]
    confidence = probs[pred_idx]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Primary Diagnosis")
        st.metric(label="Predicted Condition", value=pred_label)
        st.write(f"**Confidence:** `{confidence:.1%}`")
        if confidence < 0.40:
            st.warning("⚠️ Low model confidence. Significant biomarker overlap detected.")
            
    with col2:
        st.subheader("Differential Probability Spectrum")
        chart_df = pd.DataFrame({'Condition': classes, 'Probability': probs}).sort_values(by='Probability', ascending=True)
        fig = px.bar(chart_df, x='Probability', y='Condition', orientation='h', text_auto='.1%')
        st.plotly_chart(fig, use_container_width=True)
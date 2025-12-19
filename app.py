
import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("HR_Attrition_.pkl")


# Title
st.title("HR Attrition Prediction")
st.write("Enter employee details to predict attrition risk")

# ================= INPUT FIELDS =================

age = st.number_input("Age", min_value=18, max_value=65, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=30000)
distance_from_home = st.number_input("Distance From Home (KM)", min_value=0, max_value=100, value=10)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1,2,3,4])
work_life_balance = st.selectbox("Work Life Balance (1=Bad, 4=Excellent)", [1,2,3,4])
overtime = st.selectbox("OverTime", ["Yes", "No"])

overtime = 1 if overtime == "Yes" else 0

if st.button("Predict Attrition"):
    
    X = np.array([[age,
                   monthly_income,
                   distance_from_home,
                   years_at_company,
                   job_satisfaction,
                   work_life_balance,
                   overtime]])
    
    
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    
    if prediction == 1:
        st.error(f"⚠ Employee is likely to leave (Attrition Risk: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Employee is likely to stay (Attrition Risk: {prob*100:.2f}%)")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
scaler=joblib.load('scaler.pkl')
model=joblib.load('KNN_heart.pkl') 
expected_columns=joblib.load('columns.pkl') 

st.title("Heart Stroke Prediction App using Machine Learning")
st.markdown("This app predicts the likelihood of a heart stroke based on various health parameters. Please fill in the details below to get your prediction.")

age=st.slider("Age",18 , 100, 30 )
sex=st.selectbox("SEX",["M","F"])
chest_pain_type=st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resting_blood_pressure=st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol=st.number_input("Cholesterol", 100, 600, 200)
fasting_blood_sugar=st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
rest_ecg=st.selectbox("Resting ECG",["Normal","ST","LVH"] )
max_hr=st.slider("Max Heart Rate", 60, 220, 150)
exercise_induced_angina=st.selectbox("Exercise Induced Angina",[0,1])
oldpeak=st.slider("Oldpeak", 0.0, 6.0, 1.0)   
st_slope=st.selectbox("ST Slope",["Up","Flat","Down"])  

if st.button("Predict"):
    input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain_type],
    'RestingBP': [resting_blood_pressure],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_blood_sugar],
    'RestingECG': [rest_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': ['Y' if exercise_induced_angina == 1 else 'N'],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
})
    input_data = pd.get_dummies(input_data)
    for col in expected_columns:    
        if col not in input_data.columns:
            input_data[col] = 0         
    input_data = input_data[expected_columns]
    input_data_scaled = scaler.transform(input_data)    
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:      

        st.error("High risk of heart stroke. Please consult a doctor immediately.")
    else:       
        st.success("Low risk of heart stroke. Keep maintaining a healthy lifestyle!")   



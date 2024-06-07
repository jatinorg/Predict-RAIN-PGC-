import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('best_model.pkl')

st.title("Rainfall Prediction in Sydney")

st.write("""
## Predict whether it will rain tomorrow based on today's weather conditions.
""")

def user_input_features():
    MinTemp = st.number_input('MinTemp', min_value=-10.0, max_value=50.0, step=0.1)
    MaxTemp = st.number_input('MaxTemp', min_value=-10.0, max_value=50.0, step=0.1)
    Rainfall = st.number_input('Rainfall', min_value=0.0, max_value=500.0, step=0.1)
    Evaporation = st.number_input('Evaporation', min_value=0.0, max_value=50.0, step=0.1)
    Sunshine = st.number_input('Sunshine', min_value=0.0, max_value=15.0, step=0.1)
    Humidity9am = st.number_input('Humidity9am', min_value=0, max_value=100, step=1)
    Humidity3pm = st.number_input('Humidity3pm', min_value=0, max_value=100, step=1)
    Pressure9am = st.number_input('Pressure9am', min_value=900.0, max_value=1100.0, step=0.1)
    Pressure3pm = st.number_input('Pressure3pm', min_value=900.0, max_value=1100.0, step=0.1)
    Cloud9am = st.number_input('Cloud9am', min_value=0, max_value=8, step=1)
    Cloud3pm = st.number_input('Cloud3pm', min_value=0, max_value=8, step=1)
    Temp9am = st.number_input('Temp9am', min_value=-10.0, max_value=50.0, step=0.1)
    Temp3pm = st.number_input('Temp3pm', min_value=-10.0, max_value=50.0, step=0.1)
    RainToday = st.selectbox('RainToday', ('No', 'Yes'))
    data = {
        'MinTemp': MinTemp,
        'MaxTemp': MaxTemp,
        'Rainfall': Rainfall,
        'Evaporation': Evaporation,
        'Sunshine': Sunshine,
        'Humidity9am': Humidity9am,
        'Humidity3pm': Humidity3pm,
        'Pressure9am': Pressure9am,
        'Pressure3pm': Pressure3pm,
        'Cloud9am': Cloud9am,
        'Cloud3pm': Cloud3pm,
        'Temp9am': Temp9am,
        'Temp3pm': Temp3pm,
        'RainToday': 1 if RainToday == 'Yes' else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input parameters')
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write('Rain Tomorrow' if prediction[0] == 1 else 'No Rain Tomorrow')

st.subheader('Prediction Probability')
st.write(prediction_proba)

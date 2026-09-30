import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Machine Failure Prediction", page_icon="⚙️")

st.title("⚙️ Industrial Machine Failure Prediction")
st.write("Enter machine operating and maintenance values to predict failure risk.")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

temperature = st.number_input("Temperature", min_value=0.0, value=70.0, step=0.1)
vibration = st.number_input("Vibration", min_value=0.0, value=4.5, step=0.1)
pressure = st.number_input("Pressure", min_value=0.0, value=100.0, step=0.1)
runtime = st.number_input("Runtime", min_value=0.0, value=500.0, step=1.0)
maintenance_gap = st.number_input("Maintenance Gap (days)", min_value=0.0, value=25.0, step=1.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "runtime": runtime,
        "maintenance_gap": maintenance_gap
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"Predicted result: FAILURE RISK")
    else:
        st.success(f"Predicted result: NO FAILURE RISK")

    st.write(f"Estimated failure probability: **{probability:.2%}**")

st.caption("Educational prototype. Predictions are not production safety decisions.")

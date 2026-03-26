import streamlit as st
import requests

FASTAPI_URL = "https://claim-prediction-api.onrender.com/predict"

st.set_page_config(page_title="Claim Prediction Dashboard", layout="wide")

st.title("Claim Prediction Dashboard")
st.write("Enter customer, vehicle, and policy details to predict claim risk.")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Home", "Prediction"])

if section == "Home":
    st.header("Welcome")
    st.markdown("""
This Streamlit app is connected to a FastAPI backend.

**Workflow**
1. User enters policy details
2. Streamlit sends data to FastAPI
3. FastAPI applies the trained model
4. Prediction is returned and displayed
""")

if section == "Prediction":
    st.header("Claim Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
        annual_premium_kes = st.number_input("Annual Premium (KES)", min_value=0.0, value=50000.0, step=1000.0)
        driver_experience_years = st.number_input("Driver Experience (Years)", min_value=0, max_value=80, value=5)
        vehicle_value_kes = st.number_input("Vehicle Value (KES)", min_value=0.0, value=1200000.0, step=10000.0)
        vehicle_age = st.number_input("Vehicle Age", min_value=0, max_value=50, value=5)
        vehicle_engine_capacity = st.number_input("Vehicle Engine Capacity (CC)", min_value=600, max_value=7000, value=1500)
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2023)

    with col2:
        third_party_only = st.selectbox("Third Party Only", ["Yes", "No"])
        use_purpose = st.selectbox("Use Purpose", ["Personal", "Commercial", "Taxi"])
        region = st.selectbox(
            "Region",
            ["Nairobi", "Mombasa", "Kisumu", "Eldoret", "Meru", "Kakamega", "Thika", "Other"]
        )
        policy_term_months = st.number_input("Policy Term (Months)", min_value=1, max_value=24, value=12)
        gender = st.selectbox("Gender", ["Male", "Female"])
        vehicle_type = st.selectbox("Vehicle Type", ["Commercial", "Private", "PSV", "Motorcycle", "Other"])

    if st.button("Predict Claim Risk"):
        payload = {
            "customer_age": customer_age,
            "annual_premium_kes": annual_premium_kes,
            "driver_experience_years": driver_experience_years,
            "vehicle_value_kes": vehicle_value_kes,
            "vehicle_age": vehicle_age,
            "vehicle_engine_capacity": vehicle_engine_capacity,
            "third_party_only": third_party_only,
            "use_purpose": use_purpose,
            "region": region,
            "policy_term_months": policy_term_months,
            "gender": gender,
            "vehicle_type": vehicle_type,
            "year": int(year)
        }

        try:
            response = requests.post(
                FASTAPI_URL,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            st.subheader("Prediction Result")
            st.success(result["label"])

            if result.get("claim_probability") is not None:
                st.metric("Claim Probability", f"{result['claim_probability']:.2%}")
            else:
                st.info("Probability not available for this model.")

            st.json(result)

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to FastAPI backend: {e}")
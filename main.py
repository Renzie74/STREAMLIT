from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Claim Prediction API")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "claim_model.pkl")
model = joblib.load(MODEL_PATH)


class ClaimInput(BaseModel):
    customer_age: int
    annual_premium_kes: float
    driver_experience_years: int
    vehicle_value_kes: float
    vehicle_age: int
    vehicle_engine_capacity: int
    third_party_only: str
    use_purpose: str
    region: str
    policy_term_months: int
    gender: str
    vehicle_type: str
    year: int


@app.get("/")
def root():
    return {"message": "Claim Prediction API is running"}


@app.post("/predict")
def predict(data: ClaimInput):
    try:
        input_df = pd.DataFrame([{
            "Customer_Age": data.customer_age,
            "Annual_Premium_KES": data.annual_premium_kes,
            "Driver_Experience_Years": data.driver_experience_years,
            "Vehicle_Value_KES": data.vehicle_value_kes,
            "Vehicle_Age": data.vehicle_age,
            "Vehicle_Engine_Capacity": data.vehicle_engine_capacity,
            "Third_Party_Only": data.third_party_only,
            "Use_Purpose": data.use_purpose,
            "Region": data.region,
            "Policy_Term_Months": data.policy_term_months,
            "Gender": data.gender,
            "Vehicle_Type": data.vehicle_type,
            "Year": data.year
        }])

        print("Input columns:", input_df.columns.tolist())
        print("Input values:", input_df.iloc[0].to_dict())

        prediction = model.predict(input_df)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])

        label = "Claim Likely" if int(prediction) == 1 else "No Claim Likely"

        return {
            "prediction": int(prediction),
            "label": label,
            "claim_probability": probability
        }

    except Exception as e:
        print("PREDICTION ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
# src/serving/app.py

""" FastAPI...Interacts with our Model"""

import pandas as pd
import numpy as np
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import os
import joblib

# ---  Step 1: Defining FastAPI
app=FastAPI(
    title="Churn Prediction API",
    description="API for Customer Churn Prediction",
    version="1.0.0"
)

# ---  Step 2: Loading MOdel and Scaler
MODEL_PATH="models/LogisticRegression.pkl"
SCALER_PATH="models/scaler.pkl"

# checking if file exists
if not os.path.exists(MODEL_PATH):
    raise FileExistsError(f"Model not found at :{MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileExistsError(f"Scaler not found at:{SCALER_PATH}")

# Loading
model=joblib.load(MODEL_PATH)
scaler=joblib.load(SCALER_PATH)

print("Model is Loaded.")
print("Scaler is Loaded.")

# ---  Step 3: Defining Data format for Input
class CustomerData(BaseModel):
    gender                                  : int
    SeniorCitizen                           : int
    Partner                                 : int
    Dependents                              : int
    tenure                                  : float
    PhoneService                            : int
    MultipleLines                           : int
    OnlineSecurity                          : int
    OnlineBackup                            : int
    DeviceProtection                        : int
    TechSupport                             : int
    StreamingTV                             : int
    StreamingMovies                         : int
    PaperlessBilling                        : int
    MonthlyCharges                          : float
    TotalCharges                            : float
    InternetService_Fiber_optic             : int
    InternetService_No                      : int
    Contract_One_year                       : int
    Contract_Two_year                       : int
    PaymentMethod_Credit_card_automatic     : int
    PaymentMethod_Electronic_check          : int
    PaymentMethod_Mailed_check              : int


# ---  Step 4: Define Routes
# Route 1: Homepage
@app.get("/")
def home():
    return{
        "message":"Churn Prediction API is running",
        "status":"helathy",
        "version":"1.0.0"
    }

# Route 2: Health check
@app.get("/health")
def health_check():
    return{
        "status":"healthy",
        "model":"LogisticRegression",
        "loaded":True
    }

# Route 3: Prediction
@app.post("/predict")
def predict(customer: CustomerData):
    """
    Take customer data and predict whether he/she will churn or not 
    """

    try:
        # converting customer data to dictionary
        data=customer.dict()
        
        # converting into dataframe format
        df=pd.DataFrame([data])

        # scaling numeric columns
        cols_to_scale=['tenure','MonthlyCharges','TotalCharges']
        df[cols_to_scale]=scaler.transform(df[cols_to_scale])

        # prediction
        prediction=model.predict(df)[0]
        probability=model.predict_proba(df)[0]

        # preparing the result
        churn_prob=round(float(probability[1])*100,2)

        result={
            "churn_prediction" : int(prediction),
            "churn_label"      : "Yes" if prediction==1 else "No",
            "churn_probability": churn_prob,
            "message"          : (
                "Customer is about to leave!"
                if prediction==1
                else
                "Customer is OK"
            )
        }
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Prediction: {str(e)}"
        )
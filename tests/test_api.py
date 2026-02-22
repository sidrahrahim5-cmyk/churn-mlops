# tests/test_api.py

from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.serving.app import app

client = TestClient(app)


def test_home():
    """Home route test"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    print("Home test pass!")


def test_health():
    """Health check test"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("Health test pass!")


def test_predict_churn():
    """High risk customer — should churn"""
    data = {
        "gender": 0,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 2,
        "PhoneService": 1,
        "MultipleLines": 0,
        "OnlineSecurity": 0,
        "OnlineBackup": 0,
        "DeviceProtection": 0,
        "TechSupport": 0,
        "StreamingTV": 0,
        "StreamingMovies": 0,
        "PaperlessBilling": 1,
        "MonthlyCharges": 70.0,
        "TotalCharges": 150.0,
        "InternetService_Fiber_optic": 1,
        "InternetService_No": 0,
        "Contract_One_year": 0,
        "Contract_Two_year": 0,
        "PaymentMethod_Credit_card_automatic": 0,
        "PaymentMethod_Electronic_check": 1,
        "PaymentMethod_Mailed_check": 0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert "churn_probability" in response.json()
    print("Churn prediction test pass!")


def test_predict_no_churn():
    """Loyal customer — should not churn"""
    data = {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 1,
        "tenure": 60,
        "PhoneService": 1,
        "MultipleLines": 1,
        "OnlineSecurity": 1,
        "OnlineBackup": 1,
        "DeviceProtection": 1,
        "TechSupport": 1,
        "StreamingTV": 1,
        "StreamingMovies": 1,
        "PaperlessBilling": 0,
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
        "InternetService_Fiber_optic": 0,
        "InternetService_No": 0,
        "Contract_One_year": 0,
        "Contract_Two_year": 1,
        "PaymentMethod_Credit_card_automatic": 1,
        "PaymentMethod_Electronic_check": 0,
        "PaymentMethod_Mailed_check": 0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    print("No-churn prediction test pass!")
# src/data/clean_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

if __name__ == "__main__":

    # --- 1: Load
    df = pd.read_csv("data/raw/churn_data.csv")
    print("Loaded:", df.shape)

    # --- 2: Drop customerID
    df = df.drop(columns=['customerID'])
    print("customerID dropped")

    # --- 3: Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    )
    df['TotalCharges'] = df['TotalCharges'].fillna(
        df['TotalCharges'].median()
    )
    print("TotalCharges fixed")

    # --- 4: Binary columns
    yes_no_cols = [
        'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling', 'Churn'
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    print("Binary columns encoded")

    # --- 5: Service columns
    service_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        df[col] = df[col].replace('No internet service', 'No')
        df[col] = df[col].replace('No phone service', 'No')
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    print("Service columns encoded")

    # --- 6: One-Hot Encoding
    df = pd.get_dummies(
        df,
        columns=['InternetService', 'Contract', 'PaymentMethod'],
        drop_first=True
    )

    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    print("Column names:", list(df.columns))

    print("One-hot encoding done")
    print("Shape after encoding:", df.shape)

    # --- 7: PEHLE SPLIT
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print("Split done")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    # --- 8: PHIR SCALE
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(
        X_train[cols_to_scale]
    )
    X_test[cols_to_scale] = scaler.transform(
        X_test[cols_to_scale]
    )
    print("Scaling done")

    # --- 9: Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved")

    # --- 10: Save data
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv",   index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv",   index=False)

    print("\n✅ Sab kuch save ho gaya!")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
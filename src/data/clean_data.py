# src/data/clean_data.py

'''
Data Cleaning scipt
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

# --- Step 1: Loading Data
def load_raw_data(file_path):
    print("=" * 45)
    print(" Raw Data is loading...")
    print("=" * 45)

    df=pd.read_csv(file_path)
    print(f" Shape: {df.shape}")
    return df

# --- Step 2: Removing unnecessary columns
def drop_unnecessary_columns(df):
    print("\nRemoving Unnecessary columns")
    df=df.drop(columns=['customerID'])
    print("Removed CustomerID")
    return df

# --- Step 3: Fixing TotalCharges
def fix_total_charges(df):
    """Changing Totalcharges to float from Text."""
    print("\n TotalCharges is fixing...")
    df['TotalCharges']=pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    nan_count=df['TotalCharges'].isnull().sum() # finding NaN values
    print(f" NaN values are:{nan_count}")

    median_val=df['TotalCharges'].median()
    df['TotalCharges']=df['TotalCharges'].fillna(median_val) # fillingup NaN with median values
    print(f" Median is filled: {median_val:.2f}")

    return df

# --- Step 4: Fixing Binary columns
def encode_binary_columns(df):
    """Changing columns having two values to 0/1"""
    print("\n Binary columns is being encoded...")
    yes_no_cols=[
        'Partner','Dependents','PhoneService',
        'PaperlessBilling','Churn'
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        print(f"{col}: Yes=1, No=0")

    # Female=1, Male=0
    df['gender']=df['gender'].map({'Female':1,'Male':0})
    print(f"gender:Female=1,Male=0")

    return df

# --- Step 5: Fixing Multi-category columns
def encode_categorical_columns(df):
    """Handling columns with 3+ categories by one-hot encoding"""
    print("\n Categorical columns are encoding...")
    service_cols=[
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]

    for col in service_cols:
        df[col]=df[col].replace('No internet service','No')
        df[col]=df[col].replace('No phone service','No')
        df[col]=df[col].map({'Yes':1,'No':0})
        print(f"{col}: Column is encoded")

    # One-hot encoding
    ohe_cols=['InternetService','Contract','PaymentMethod']

    df=pd.get_dummies(df,columns=ohe_cols,drop_first=True)
    print(f" One-Hot Encoding is done:{ohe_cols}")

    return df

# --- Step 6: Scaling Features
def scale_features(df):
    """ Tenure and Monthly CHarges and TotalCHarges fields need to be scaled"""
    print(f"\n Features are scaling...")
    cols_to_scale=['tenure','MonthlyCharges','TotalCharges']

    scaler=StandardScaler()
    df[cols_to_scale]=scaler.fit_transform(df[cols_to_scale])
    print(f" Scaled Columns: {cols_to_scale}")

    # saving Scaler, it will be reused in API
    os.makedirs("models",exist_ok=True)
    joblib.dump(scaler,"models/scaler.pkl")
    print(" Scaler is saved at: models/scaler.pkl")
    return df, scaler

# --- Step 7: Split into Train/Test
def split_data(df):
    print("\n Data is splitting...")

    X=df.drop(columns=['Churn']) # features visible to model
    y=df['Churn'] # target variable

    print(f" Features (X):{X.shape}")
    print(f" Target (y):{y.shape}")

    X_train, X_test, y_train, y_test=train_test_split(
        X,y,
        test_size=0.2,
        random_state=42,
        stratify=y # maintaining class balance
    )
    print(f" Train size:{X_train.shape[0]} rows")
    print(f" Test size: {X_test.shape[0]} rows")

    return X-X_train, X_test, y_train, y_test

# --- Step 8: Saving Clean Data
def save_clean_data(X_train, X_test,y_train,y_test):
    print("\n Saving cleaned data...")
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv",index=False)
    X_test.to_csv("data/processed/X_test.csv",index=False)
    y_train.to_csv("data/processed/y_train.csv",index=False)
    y_test.to_csv("data/processed/y_test.csv",index=False)

    print(" X_train.csv saved")
    print(" X_test.csv saved")
    print(" y_train.csv saved")
    print(" y_test.csv saved")

# --- Main
if __name__=="__main__":
    DATA_PATH="data/raw/churn_data.csv"
    if not os.path.exists(DATA_PATH):
        print("Data file is not found.")
        exit()

    df=load_raw_data(DATA_PATH)
    df=drop_unnecessary_columns(df)
    df=fix_total_charges(df)
    df=encode_binary_columns(df)
    df=encode_categorical_columns(df)
    df,scaler=scale_features(df)

    X_train,X_test,y_train,y_test=split_data(df)
    save_clean_data(X_train,X_test,y_train,y_test)

    print("\n" + "=" * 45)
    print(" Data Cleaning is Done!")
    print(f" Total features: {X_train.shape[1]}")
    print("=" * 45)
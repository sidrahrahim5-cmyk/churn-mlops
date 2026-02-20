# src/data/explore_data.py

"""
EDA _ Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Step 1: Download Data ---
def load_data(file_path):
    print("="*45)
    print(" Data is downloading...")
    print("="*45)

    df=pd.read_csv(file_path)

    print(f" Rows     : {df.shape[0]}")
    print(f" Columns  : {df.shape[1]}")

    return df

# --- Step 2: Exploring Basic Info of Data ---
def basic_info(df):
    print("\n"+"="*45)
    print(" Basic Information")
    print("="*45)

    print("First 5 Rows:")
    print(df.head())

    print("\n Column Types:")
    print(df.dtypes)

    print("\n Column Values:")
    missing =df.isnull().sum()
    print(missing[missing>0])

    if missing.sum()==0:
        print("No missing value")

# --- Step 3: Exploring Target Data ---
def analyze_target(df):
    print("\n"+"="*45)
    print(" Target - Churn Analysis")
    print("="*45)

    counts=df['Churn'].value_counts()
    percent=df['Churn'].value_counts(normalize=True)* 100

    print(f"\nCustomer who didn't leave (No) : {counts['No']}  ({percent['No']:.1f}%)")
    print(f"Customers who left    (Yes) : {counts['Yes']} ({percent['Yes']:.1f}%)")

    # Visulaization of data
    os.makedirs("data/processed",exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4))

    # Bar Chart
    counts.plot(kind='bar',ax=axes[0],
                color=['green','red'])
    axes[0].set_title('Churn Count')
    axes[0].set_xlabel('Churn')
    axes[0].set_ylabel('Churn')
    axes[0].tick_params(axis='x',rotation=0)

    # Pie Chart
    percent.plot(kind='pie', ax=axes[1],
                autopct='%1.1f%%',
                colors=['green','red'])
    axes[1].set_title('Churn Percentage')

    plt.tight_layout()
    plt.savefig('data/processed/churn_distribution.png')
    print("\n Graph is saved at: data/processed/churn_distribution.png")
    plt.show()

# --- Step 4: Exploring Numerical Feature ---
def analyze_numeric(df):
        print("\n"+"="*45)
        print(" Numerical Features Analysis")
        print("="*45)

        # Transform TotalCharges
        df['TotalCharges']=pd.to_numeric(
            df['TotalCharges'],errors='coerce'
        )

        numeric_cols=['tenure','MonthlyCharges','TotalCharges']

        fig, axes=plt.subplots(2,3,figsize=(14,8))

        for idx, col in enumerate(numeric_cols):
            axes[0,idx].hist(
                df[col].dropna(),
                bins=30,
                color='steelblue',
                edgecolor='white'
            )
            axes[0,idx].set_title(f'{col} - Distribution')

            no_churn=df[df['Churn']=='No'][col].dropna()
            yes_churn=df[df['Churn']=='Yes'][col].dropna()

            axes[1,idx].hist(no_churn,bins=30,
                             alpha=0.6, color='green',
                             label='No Churn')
            axes[1, idx].hist(yes_churn,bins=30,
                              alpha=0.6, color='red',
                              label='Churn')
            axes[1,idx].set_title(f"{col} - by Churn")
            axes[1,idx].legend()

        plt.tight_layout()
        plt.savefig('data/processed/numeric_features.png')
        plt.show()

# --- Step 5: Exploring Categorical Features ---
def analyze_categorical(df):
    print("\n"+"="*45)
    print(" Categorical Features Analysis")
    print("="*45)

    cat_cols=[
        'Contract',
        'PaymentMethod',
        'InternetService',
        'gender'
    ]

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes=axes.flatten()

    for idx,col in enumerate(cat_cols):

        churn_rate=df.groupby(col)['Churn'].apply(
            lambda x:(x=="Yes").sum()/len(x)*100
        ).sort_values(ascending=False)

        churn_rate.plot(kind='bar',ax=axes[idx],
                        color='coral',
                        edgecolor='white')
        axes[idx].set_title(f"{col} - Churn %")
        axes[idx].set_ylabel('Churn %')
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('data/processed/categorical_features.png')
    plt.show()

#  Main 
if __name__=="__main__":
    DATA_PATH="data/raw/churn_data.csv"

    if not os.path.exists(DATA_PATH):
        print("No Data file found!")
        print("First execute download_data.py")
        exit()

    df=load_data(DATA_PATH)
    basic_info(df)
    analyze_target(df)
    analyze_numeric(df)
    analyze_categorical(df)

    print("\n" + "=" *45)
    print("EDA completed")
    print("=" *45)
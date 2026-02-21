# src/models/train.py

"""
Model Training with MLFlow Tracking
Used Random Forest Model Training
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import mlflow
import mlflow.sklearn
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

#  ---  Step 1: Loading Data
def load_data():
    print("=" * 45)
    print("Training Data is loading...")
    print("=" * 45)

    X_train=pd.read_csv("data/processed/X_train.csv")
    X_test=pd.read_csv("data/processed/X_test.csv")
    y_train=pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test=pd.read_csv("data/processed/y_test.csv").squeeze()

    print(f" X_train: {X_train.shape}")
    print(f" X_test: {X_test.shape}")
    return X_train,X_test,y_train,y_test

#  ---  Step 2: Calculate Metrics
def calculate_metrics(y_true,y_pred):
    """
    Metrics:
    Accuracy 
    Precision
    Recall
    F1 Score
    """
    metrics={
        'accuracy':accuracy_score(y_true,y_pred),
        'precision': precision_score(y_true,y_pred),
        'recall':recall_score(y_true,y_pred),
        'f1_score':f1_score(y_true,y_pred)
    }
    return metrics

#  ---  Step 3: Creating Confusion Matrix
def plot_confusion_matrix(y_true,y_pred,  model_name):

    cm=confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Churn','Churn'],
        yticklabels=['No Churn','Churn']
    )
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    # Save
    path=f"data/processed/{model_name}_confusion_matrix.png"
    plt.savefig(path)
    plt.show()

    return path

#  ---  Step 4:Train Model with MLflow
def train_model(model, model_name, params,
                X_train, X_test, y_train, y_test):
    '''
    This function starts MLflow, trains model, calculates metrics and log everything in MLflow
    '''

    print(f"\n{'=' * 45}")
    print(f" {model_name} Train is started...")
    print(f"{'=' * 45}")

    # setting MLflow experiment
    mlflow.set_experiment('churn-prediction')

    # starting MLflow run (an experiment session)
    with mlflow.start_run(run_name=model_name):

        # fillup NaN values
        X_train = X_train.fillna(0)
        X_test  = X_test.fillna(0)

        # training model
        model.fit(X_train,y_train)
        print(f"Model is trained.")

        # predictions
        y_pred=model.predict(X_test)

        #calculating metrics
        metrics=calculate_metrics(y_test,y_pred)

        print(f"\n Results:")
        print(f" Accuracy : {metrics['accuracy']:.4f}")
        print(f" Precision: {metrics['precision']:.4f}")
        print(f" Recall   : {metrics['recall']:.4f}")
        print(f" F1 Score : {metrics['f1_score']:.4f}")

        # --- Logging in MLflow ---
        # Logging Parameters
        mlflow.log_params(params)

        # Logging Metrics
        mlflow.log_metrics(metrics)

        # Making confusion matrix
        cm_path=plot_confusion_matrix(
            y_test,y_pred,model_name
        )

        # Saving image in MLflow
        mlflow.log_artifact(cm_path)

        # Saving model in MLflow
        mlflow.sklearn.log_model(model,model_name)

        # Saving Model file
        os.makedirs("models",exist_ok=True)
        model_path=f"models/{model_name}.pkl"
        joblib.dump(model,model_path)
        print(f"Model saved at : {model_path}")

        # Print Classification report
        print(f"\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['No Churn', 'Churn']
        ))

    return metrics

# --- Main
if __name__ == "__main__":

    # Load Data
    X_train, X_test, y_train, y_test = load_data()

    # ----------------------------------------
    # Model 1: Logistic Regression
    # Simple model — for baseline
    # ----------------------------------------
    lr_params = {
        'model_type': 'LogisticRegression',
        'max_iter'  : 1000,
        'C'         : 1.0
    }

    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42
    )

    lr_metrics = train_model(
        model      = lr_model,
        model_name = "LogisticRegression",
        params     = lr_params,
        X_train    = X_train,
        X_test     = X_test,
        y_train    = y_train,
        y_test     = y_test
    )

    # ----------------------------------------
    # Model 2: Random Forest
    # More powerful model
    # ----------------------------------------
    rf_params = {
        'model_type' : 'RandomForest',
        'n_estimators': 100,
        'max_depth'  : 10,
        'random_state': 42
    }

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    rf_metrics = train_model(
        model      = rf_model,
        model_name = "RandomForest",
        params     = rf_params,
        X_train    = X_train,
        X_test     = X_test,
        y_train    = y_train,
        y_test     = y_test
    )

    # ----------------------------
    # Comparing both Models 
    # -----------------------------
    print("\n" + "=" * 45)
    print(" Models Comparison")
    print("=" * 45)
    print(f"\n{'Metric':<12} {'LogReg':>10} {'RandomForest':>14}")
    print("-" * 38)

    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        print(
            f"{metric:<12} "
            f"{lr_metrics[metric]:>10.4f} "
            f"{rf_metrics[metric]:>14.4f}"
        )

    print("\n" + "=" * 45)
    print(" Training is Done!")
    print(" MLflow UI :")
    print("=" * 45)
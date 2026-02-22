# Churn Prediction MLOps Project

An end-to-end Machine Learning pipeline to predict 
customer churn for a telecom company.

## Project Overview

This project predicts whether a customer will leave 
the company (churn) or not, using their service usage 
and billing data.

## Tech Stack

- **Python 3.11**
- **Scikit-learn** — ML Model
- **MLflow** — Experiment Tracking
- **FastAPI** — REST API
- **Docker** — Containerization
- **GitHub Actions** — CI/CD Pipeline

## Project Structure
```
churn-mlops/
├── src/
│   ├── data/          # Data download & cleaning
│   ├── models/        # Model training
│   ├── serving/       # FastAPI
│   └── monitoring/    # Model monitoring
├── data/
│   ├── raw/           # Raw data
│   └── processed/     # Clean data
├── models/            # Saved models
├── tests/             # Unit tests
├── Dockerfile
└── .github/workflows/ # CI/CD
```

## ML Pipeline
```
Raw Data → EDA → Cleaning → Training → API → Docker
```

## Model Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 80.55% | 65.82% | 55.61% | 60.29% |
| Random Forest | 79.99% | 65.65% | 51.60% | 57.78% |

**Best Model: Logistic Regression**

## How to Run

### Local Setup
```bash
# Clone karo
git clone https://github.com/sidrahrahim5-cmyk/churn-mlops.git
cd churn-mlops

# Virtual environment
python -m venv venv
venv\Scripts\activate

# Libraries install
pip install -r requirements.txt

# Data download
python src/data/download_data.py

# Data clean
python src/data/clean_data.py

# Model train
python src/models/train.py

# API start
uvicorn src.serving.app:app --reload
```

### Docker
```bash
# Build
docker build -t churn-mlops:v2 .

# Run
docker run -p 8000:8000 churn-mlops:v2
```

### API Test

Open browser:
```
http://127.0.0.1:8000/docs
```

### Sample Prediction
```json
POST /predict
{
  "gender": 0,
  "SeniorCitizen": 0,
  "Partner": 1,
  "tenure": 2,
  "MonthlyCharges": 70.0,
  "TotalCharges": 150.0,
  "Contract_One_year": 0,
  "Contract_Two_year": 0
}
```

Response:
```json
{
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 70.43,
  "message": "Customer is about to leave!"
}
```

## Tests
```bash
pytest tests/test_api.py -v
```

## CI/CD

Every push to main branch automatically:
- Downloads data
- Cleans data
- Trains model
- Runs tests

## Author

**Sidrah Rahim**  
[GitHub](https://github.com/sidrahrahim5-cmyk)

---
*Built with ❤️ to learn MLOps*
# 🚀 NASA NEO Risk Prediction AI

This project demonstrates how to fetch real **Near-Earth Object (NEO)** data from NASA’s API, preprocess it, and train machine learning models (Random Forest, Gradient Boosting, etc.) to predict asteroid **hazard risk scores**.

---

## 📌 Features

- Fetch asteroid data from the **NASA NEO API**.
- Preprocess raw JSON into structured features.
- Train multiple ML models (Random Forest, Gradient Boosting).
- Evaluate models with **MAE, RMSE, R²**.
- Save trained models with `joblib`.
- Serve predictions using **FastAPI + Swagger UI**.

---

## ⚙️ Setup

### 1. Clone repo and install dependencies

```bash
git clone https://github.com/waqarali5498/nsa-hackathon-ml-model
cd nsa-hackathon-ml-model
python -m venv venv
source venv/bin/activate     # On Mac/Linux
venv\Scripts\activate        # On Windows

pip install -r requirements.txt
```

### 2. Get NASA API Key

Go to: https://api.nasa.gov/

Generate a free API key.

Set it as an environment variable:

```bash
Mac/Linux:
export NASA_API_KEY="YOUR_KEY_HERE"

Windows (PowerShell):
setx NASA_API_KEY "YOUR_KEY_HERE"

```

(Restart terminal after setting this.)

### 📥 Fetching NEO Data

Run:

```bash
python fetch_data.py --start_date 2024-01-01 --end_date 2024-01-07
```

This will save asteroid data into:

data/neo_data.csv

### 🧠 Training ML Models

Run:

```bash
python train_model.py
```

Example output:

```bash
RandomForest     | MAE: 9504.88 | RMSE: 363066.06 | R²: 0.731
GradientBoosting | MAE: 9072.66 | RMSE: 320519.61 | R²: 0.790


The best model is saved to:

models/neo_model_v1.joblib
```

### 🌐 Running API (FastAPI + Swagger)

Start the API:

```bash
uvicorn app:app --reload
```

Open Swagger UI:
http://127.0.0.1:8000/docs

```bash
Test predictions with JSON input:

{
  "diameter_km": 0.5,
  "velocity_kms": 12.3,
  "miss_distance_km": 150000,
  "absolute_magnitude": 22.1
}
```

### 📊 Evaluation Metrics

```bash
MAE (Mean Absolute Error) → Lower is better.

RMSE (Root Mean Squared Error) → Penalizes large errors.

R² (Coefficient of Determination) → Closer to 1 is better.
```

### 📂 Project Structure

```bash

nsa-hackathon-ml-model/
│── data/                 # Raw and processed NEO data
│── models/               # Saved ML models (.joblib)
│── app.py                # FastAPI app for predictions
│── fetch_data.py         # Script to fetch NASA NEO data
│── train_model.py        # Train and evaluate ML models
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

### 🙌 Acknowledgements

```bash

NASA Open APIs
 for providing NEO datasets.

Scikit-learn for ML models.

FastAPI + Swagger UI for serving predictions.

```

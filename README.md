# 🚀 NASA Near-Earth Object (NEO) Risk Predictor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**An Artificial Intelligence early-warning system that automatically pulls live asteroid data directly from NASA and predicts hazard risk scores.**

---

## 🌍 The Big Picture (What is this?)
Every day, thousands of asteroids and comets—known as Near-Earth Objects (NEOs)—orbit the sun and pass relatively close to our planet. While most are harmless, keeping track of which ones pose a genuine threat requires analyzing massive amounts of complex space data.

This project automatically pulls live asteroid data directly from NASA, studies the patterns, and uses Machine Learning to predict a **Hazard Risk Score** for any given space rock based on its size, speed, and distance from Earth.

## 💡 The Value (Why does this matter?)
* **Automated Threat Detection:** Instead of humans manually crunching numbers, the AI instantly flags which asteroids require closer observation.
* **Making Space Data Accessible:** NASA's raw data is incredibly dense. This project translates complex astronomical metrics into a simple, readable prediction.
* **Scalable Infrastructure:** Built with a live API, this model could easily be plugged into a public website, an alert system, or a dashboard for researchers.

## 🧠 How It Works (In Plain English)
1. **Gathering Intel:** The system connects to NASA's official database and downloads recent asteroid sightings.
2. **Learning the Patterns:** We feed this data into advanced AI models (like Random Forest and Gradient Boosting). The AI looks at historical data to learn the difference between a harmless pebble and a dangerous threat.
3. **Predicting the Future:** Once trained, you can give the AI the specs of a brand new asteroid, and it will instantly calculate its risk level.

---

## 🛠️ Technical Documentation (For Developers)

Below are the instructions to run the data pipeline, train the Machine Learning models, and start the local API server.

### 1. Setup & Installation
Clone the repository and set up your virtual environment:

```bash
git clone [https://github.com/waqarali5498/nsa-hackathon-ml-model](https://github.com/waqarali5498/nsa-hackathon-ml-model)
cd nsa-hackathon-ml-model
python -m venv venv

# Activate the virtual environment:
source venv/bin/activate      # On Mac/Linux
venv\Scripts\activate         # On Windows

pip install -r requirements.txt
2. Configure the NASA API
To fetch live data, you need a free API key from NASA Open APIs. Set it as an environment variable:

Mac/Linux:

Bash
export NASA_API_KEY="YOUR_KEY_HERE"
Windows (PowerShell):

Bash
setx NASA_API_KEY "YOUR_KEY_HERE"
(Restart your terminal after setting the key.)

3. Fetching the Data
Run the following script to download asteroid data for a specific date range. This saves the raw data to data/neo_data.csv.

Bash
python fetch_data.py --start_date 2024-01-01 --end_date 2024-01-07
4. Training the AI Models
Execute the training script to process the data and evaluate different machine learning algorithms:

Bash
python train_model.py
Example Output:

RandomForest     | MAE: 9504.88 | RMSE: 363066.06 | R²: 0.731
GradientBoosting | MAE: 9072.66 | RMSE: 320519.61 | R²: 0.790

The best-performing model is automatically saved to: models/neo_model_v1.joblib

(Note on Metrics: MAE and RMSE measure error—lower is better. R² measures accuracy—closer to 1 is better).

5. Running the API
Start the FastAPI server to serve predictions locally:

Bash
uvicorn app:app --reload
Once running, open the interactive Swagger UI in your browser: http://127.0.0.1:8000/docs

Test a prediction with JSON:

JSON
{
  "diameter_km": 0.5,
  "velocity_kms": 12.3,
  "miss_distance_km": 150000,
  "absolute_magnitude": 22.1
}
📂 Project Structure
data/ — Raw and processed NEO data sets.

models/ — Saved Machine Learning models (.joblib).

app.py — The FastAPI application for serving predictions.

fetch_data.py — Script to interface with the NASA NEO API.

train_model.py — Script for training and evaluating the ML models.

🙌 Acknowledgements
NASA Open APIs for providing the Near-Earth Object datasets.

Scikit-learn for the machine learning architecture.

FastAPI + Swagger UI for the rapid API deployment.

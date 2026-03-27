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

## 💻 Developer Quick Start

### 1. Installation & Environment
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/waqarali5498/nsa-hackathon-ml-model](https://github.com/waqarali5498/nsa-hackathon-ml-model)
cd nsa-hackathon-ml-model
python -m venv venv
source venv/bin/activate  # Mac/Linux (Use `venv\Scripts\activate` on Windows)
pip install -r requirements.txt
```

### 2. NASA API Configuration
Get your free API key from [NASA Open APIs](https://api.nasa.gov/) and set it as an environment variable:
```bash
export NASA_API_KEY="YOUR_KEY_HERE"  # Mac/Linux
setx NASA_API_KEY "YOUR_KEY_HERE"    # Windows (Restart terminal after)
```

### 3. Data Ingestion & Model Training
First, download the latest asteroid data, then train the Machine Learning models:
```bash
python fetch_data.py --start_date 2024-01-01 --end_date 2024-01-07
python train_model.py
```

**Model Performance Snapshot:**
| Model Architecture | MAE (Error) ↓ | RMSE ↓ | R² (Accuracy) ↑ |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | 9,072.66 | 320,519.61 | **0.790** |
| **Random Forest** | 9504.88 | 363,066.06 | 0.731 |

*The best model is automatically serialized and saved to `models/neo_model_v1.joblib`.*

### 4. Running the API
Launch the FastAPI server to serve local predictions:
```bash
uvicorn app:app --reload
```
Navigate to `http://127.0.0.1:8000/docs` to use the interactive Swagger UI, or test the endpoint directly with this JSON payload:

```json
{
  "diameter_km": 0.5,
  "velocity_kms": 12.3,
  "miss_distance_km": 150000,
  "absolute_magnitude": 22.1
}
```

---

## 📂 Project Architecture

```text
📦 nsa-hackathon-ml-model
 ┣ 📂 data           # Raw & processed NEO datasets (CSV)
 ┣ 📂 models         # Serialized ML models (.joblib)
 ┣ 📜 app.py         # FastAPI prediction server & endpoints
 ┣ 📜 fetch_data.py  # NASA API ingestion and preprocessing script
 ┣ 📜 train_model.py # Model training, evaluation, and serialization
 ┗ 📜 requirements.txt
```

---

## 🙌 Acknowledgements
* **[NASA Open APIs](https://api.nasa.gov/)** for providing the Near-Earth Object datasets.
* **[Scikit-learn](https://scikit-learn.org/)** for the machine learning architecture.
* **[FastAPI](https://fastapi.tiangolo.com/) + Swagger UI** for the rapid API deployment.
```

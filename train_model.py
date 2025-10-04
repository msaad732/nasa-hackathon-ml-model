# train_model.py
import os
import math
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional imports (only if installed)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


DATA_CSV = "data/neo_feed.csv"
OUT_MODEL = "models/neo_model_best.joblib"
os.makedirs("models", exist_ok=True)

# Training default density (kg/m^3)
TRAINING_DENSITY = 3000.0  # "stony" default

def energy_megatons_from_diameter_velocity(diameter_m, velocity_km_s, density=TRAINING_DENSITY):
    r = diameter_m / 2.0
    volume = (4.0/3.0) * math.pi * (r**3)           # m^3
    mass = density * volume                         # kg
    v = velocity_km_s * 1000.0                      # m/s
    energy_j = 0.5 * mass * (v**2)                  # Joules
    energy_mt = energy_j / 4.184e15                 # convert J → megatons TNT
    return energy_mt


def add_features(df):
    """Feature engineering: add derived features"""
    df["log_diameter"] = np.log1p(df["diameter_m"])
    df["velocity_sq"] = df["velocity_km_s"] ** 2
    df["inv_miss_distance"] = 1.0 / (df["miss_distance_km"] + 1.0)
    return df


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train + evaluate a model and return metrics"""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2, model


def main():
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["diameter_m", "velocity_km_s"])

    # Compute label (energy in megatons)
    df["energy_mt"] = df.apply(lambda r: energy_megatons_from_diameter_velocity(
        r["diameter_m"], r["velocity_km_s"]), axis=1)

    # Base features
    feature_cols = ["diameter_m", "velocity_km_s", "eccentricity", "miss_distance_km", "hazardous"]

    # Fill missing values with medians
    medians = df[feature_cols].median().to_dict()
    df.fillna(medians, inplace=True)

    # Add engineered features
    df = add_features(df)
    feature_cols += ["log_diameter", "velocity_sq", "inv_miss_distance"]

    X = df[feature_cols].values
    y = df["energy_mt"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Candidate models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                                         colsample_bytree=0.8, random_state=42, n_jobs=-1)
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=-1, random_state=42, n_jobs=-1)

    results = []
    best_model = None
    best_score = -np.inf

    print("\n🔹 Training & Evaluating Models...\n")
    for name, model in models.items():
        mae, rmse, r2, trained_model = evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append((name, mae, rmse, r2))
        print(f"{name:15s} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
        if r2 > best_score:
            best_score = r2
            best_model = trained_model
            best_name = name

    # Save best model
    payload = {
        "model": best_model,
        "feature_cols": feature_cols,
        "medians": medians,
        "training_density": TRAINING_DENSITY,
        "best_model_name": best_name
    }
    joblib.dump(payload, OUT_MODEL)
    print(f"\n✅ Best model: {best_name} (R²={best_score:.3f}) saved to {OUT_MODEL}")


if __name__ == "__main__":
    main()

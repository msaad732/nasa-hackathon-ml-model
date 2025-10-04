# import math
# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI()

# # density by composition
# DENSITY = {
#     "metallic": 8000,  # kg/m^3
#     "stony": 3000,
#     "icy": 1000
# }

# class ImpactParams(BaseModel):
#     diameter: float  # meters
#     velocity: float  # km/s
#     composition: str
#     location: str  # "ocean" or "land"

# @app.post("/predict")
# def predict_impact(data: ImpactParams):
#     radius = data.diameter / 2
#     density = DENSITY.get(data.composition.lower(), 3000)
#     volume = (4/3) * math.pi * (radius**3)
#     mass = volume * density
#     velocity = data.velocity * 1000  # convert km/s to m/s
#     energy_joules = 0.5 * mass * (velocity**2)
#     energy_mt = energy_joules / 4.18e15  # convert to megatons TNT

#     # hazard scoring logic
#     if energy_mt < 1:
#         hazard = "Low"
#     elif energy_mt < 10:
#         hazard = "Moderate"
#     elif energy_mt < 50:
#         hazard = "Severe"
#     else:
#         hazard = "Catastrophic"

#     # risk breakdown
#     blast_risk = min(100, int(energy_mt * 2))
#     thermal_risk = min(100, int(energy_mt * 1.5))
#     tsunami_risk = 0

#     if data.location.lower() == "ocean":
#         tsunami_risk = min(100, int(energy_mt * 3))
#         blast_risk = int(blast_risk * 0.7)
#         thermal_risk = int(thermal_risk * 0.5)

#     return {
#         "hazard_level": hazard,
#         "blast_risk": blast_risk,
#         "thermal_risk": thermal_risk,
#         "tsunami_risk": tsunami_risk,
#         "estimated_megatons": round(energy_mt, 2)
#     }


# main.py — add/modify these parts

import os
import joblib
import numpy as np
import math
from fastapi import FastAPI

# existing imports / app definition
app = FastAPI()

# densities (same as before)
DENSITY = {
    "metallic": 8000.0,
    "stony": 3000.0,
    "icy": 1000.0
}

# load ML model if available
MODEL_PATH = "models/neo_model_v1.joblib"
MODEL_PAYLOAD = None
if os.path.exists(MODEL_PATH):
    try:
        MODEL_PAYLOAD = joblib.load(MODEL_PATH)
        MODEL = MODEL_PAYLOAD["model"]
        FEATURE_COLS = MODEL_PAYLOAD["feature_cols"]
        MEDIANS = MODEL_PAYLOAD["medians"]
        TRAINING_DENSITY = MODEL_PAYLOAD.get("training_density", 3000.0)
        print("Loaded ML model:", MODEL_PATH)
    except Exception as e:
        print("Failed to load model:", e)
        MODEL = None
else:
    MODEL = None
    FEATURE_COLS = ["diameter_m","velocity_km_s","eccentricity","miss_distance_km","hazardous"]
    MEDIANS = {c: 0.1 for c in FEATURE_COLS}
    TRAINING_DENSITY = 3000.0

# physics function (same formula as training)
def physics_energy_mt(diameter_m, velocity_km_s, density):
    r = diameter_m / 2.0
    volume = (4.0/3.0) * math.pi * (r**3)
    mass = density * volume
    v = velocity_km_s * 1000.0
    energy_j = 0.5 * mass * (v**2)
    energy_mt = energy_j / 4.184e15
    return energy_mt

# risk mapping helper
def map_risks(energy_mt, location):
    # basic thresholds
    if energy_mt < 1:
        hazard = "Low"
    elif energy_mt < 10:
        hazard = "Moderate"
    elif energy_mt < 50:
        hazard = "Severe"
    else:
        hazard = "Catastrophic"

    # base scores (tunable)
    base_blast = min(100, int(energy_mt * 3))
    base_thermal = min(100, int(energy_mt * 1.5))
    base_tsunami = min(100, int(energy_mt * 4))

    if location.lower() == "ocean":
        tsunami = base_tsunami
        blast = int(base_blast * 0.4)
        thermal = int(base_thermal * 0.5)
    else:
        tsunami = int(base_tsunami * 0.05)
        blast = base_blast
        thermal = base_thermal

    # ensure 0-100
    return {
        "hazard_level": hazard,
        "blast_risk": max(0, min(100, int(blast))),
        "thermal_risk": max(0, min(100, int(thermal))),
        "tsunami_risk": max(0, min(100, int(tsunami))),
        "estimated_megatons": round(energy_mt, 2)
    }

# Your existing Pydantic model, route definitions, etc. keep unchanged.
# Replace or update the predict endpoint to use the ML model:

from pydantic import BaseModel

class ImpactParams(BaseModel):
    diameter: float  # meters
    velocity: float  # km/s
    composition: str
    location: str  # "ocean" or "land"

@app.post("/predict")
def predict_impact(data: ImpactParams):
    diameter_m = data.diameter
    velocity_km_s = data.velocity
    comp = data.composition.lower()
    location = data.location.lower()

    # step 1: try ML model prediction (uses medians for missing features)
    if MODEL is not None:
        feature_vector = []
        for col in FEATURE_COLS:
            if col == "diameter_m":
                feature_vector.append(diameter_m)
            elif col == "velocity_km_s":
                feature_vector.append(velocity_km_s)
            else:
                # use medians saved from training
                feature_vector.append(MEDIANS.get(col, 0.0))
        energy_mt_pred = float(MODEL.predict([feature_vector])[0])
        # adjust model prediction to match user composition density
        density_user = DENSITY.get(comp, 3000.0)
        energy_mt_adj = energy_mt_pred * (density_user / TRAINING_DENSITY)
        energy_mt = max(0.0, energy_mt_adj)
    else:
        # fallback to pure physics (use composition density)
        density_user = DENSITY.get(comp, 3000.0)
        energy_mt = physics_energy_mt(diameter_m, velocity_km_s, density_user)

    # map to risk percentages
    result = map_risks(energy_mt, location)
    return result

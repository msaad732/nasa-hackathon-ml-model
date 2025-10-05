import streamlit as st
import math
import joblib
import os
import impact_utils
import numpy as np

# ---- Densities ----
DENSITY = {"metallic": 8000.0, "stony": 3000.0, "icy": 1000.0}

# ---- Load ML model ----
MODEL_PATH = "models/neo_model_v1.joblib"
MODEL = None
FEATURE_COLS = ["diameter_m","velocity_km_s","eccentricity","miss_distance_km","hazardous"]
MEDIANS = {c: 0.1 for c in FEATURE_COLS}
TRAINING_DENSITY = 3000.0

if os.path.exists(MODEL_PATH):
    try:
        MODEL_PAYLOAD = joblib.load(MODEL_PATH)
        MODEL = MODEL_PAYLOAD["model"]
        FEATURE_COLS = MODEL_PAYLOAD["feature_cols"]
        MEDIANS = MODEL_PAYLOAD["medians"]
        TRAINING_DENSITY = MODEL_PAYLOAD.get("training_density", 3000.0)
    except Exception as e:
        st.warning(f"Failed to load model: {e}")

# ---- Physics energy function ----
def physics_energy_mt(diameter_m, velocity_km_s, density):
    r = diameter_m / 2.0
    volume = (4/3) * math.pi * r**3
    mass = density * volume
    v = velocity_km_s * 1000
    energy_j = 0.5 * mass * v**2
    return energy_j / 4.184e15

# ---- Risk mapping ----
def map_risks(energy_mt, location):
    if energy_mt < 1:
        hazard = "Low"
    elif energy_mt < 10:
        hazard = "Moderate"
    elif energy_mt < 50:
        hazard = "Severe"
    else:
        hazard = "Catastrophic"

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

    return {
        "hazard_level": hazard,
        "blast_risk": max(0, min(100, blast)),
        "thermal_risk": max(0, min(100, thermal)),
        "tsunami_risk": max(0, min(100, tsunami)),
        "estimated_megatons": round(energy_mt, 2)
    }

# ---- Streamlit UI ----
st.title("Asteroid Impact & Seismic Predictor 🌠")

# ---- Create tabs ----
tab1, tab2, tab3 = st.tabs(["Impact Hazard", "Seismic from Diameter", "Seismic from Mass"])

# ---- Impact Hazard Tab ----
with tab1:
    st.header("Impact Hazard Prediction")
    diameter = st.number_input("Diameter (m)", min_value=0.1, value=50.0)
    velocity = st.number_input("Velocity (km/s)", min_value=0.01, value=20.0)
    composition = st.selectbox("Composition", ["Metallic", "Stony", "Icy"])
    location = st.selectbox("Impact Location", ["Land", "Ocean"])

    if st.button("Predict Hazard", key="hazard_btn"):
        comp = composition.lower()
        location_lower = location.lower()

        # ML prediction or physics fallback
        if MODEL is not None:
            feature_vector = []
            for col in FEATURE_COLS:
                if col == "diameter_m":
                    feature_vector.append(diameter)
                elif col == "velocity_km_s":
                    feature_vector.append(velocity)
                else:
                    feature_vector.append(MEDIANS.get(col, 0.0))
            energy_mt_pred = float(MODEL.predict([feature_vector])[0])
            density_user = DENSITY.get(comp, 3000.0)
            energy_mt = max(0.0, energy_mt_pred * (density_user / TRAINING_DENSITY))
        else:
            density_user = DENSITY.get(comp, 3000.0)
            energy_mt = physics_energy_mt(diameter, velocity, density_user)

        result = map_risks(energy_mt, location_lower)
        st.json(result)

# ---- Seismic from Diameter Tab ----
with tab2:
    st.header("Seismic Impact from Diameter")
    dia = st.number_input("Diameter (m)", min_value=0.1, value=50.0, key="dia")
    vel = st.number_input("Velocity (m/s)", min_value=0.1, value=20000.0, key="vel")
    density_input = st.number_input("Density (kg/m³)", min_value=0.0, value=float(impact_utils.DEFAULT_DENSITY), key="density")

    if st.button("Predict Seismic from Diameter", key="seismic_dia_btn"):
        seismic_result = impact_utils.impact_from_diameter(
            dia,
            vel,
            density_input
        )
        st.json({
            **seismic_result,
            "description": f"The impact would generate seismic waves equivalent to a Magnitude {seismic_result['magnitude']:.2f} earthquake."
        })

# ---- Seismic from Mass Tab ----
with tab3:
    st.header("Seismic Impact from Mass")
    mass = st.number_input("Mass (kg)", min_value=0.1, value=1e9, key="mass")
    vel_mass = st.number_input("Velocity (m/s)", min_value=0.1, value=20000.0, key="vel_mass")

    if st.button("Predict Seismic from Mass", key="seismic_mass_btn"):
        seismic_mass_result = impact_utils.impact_from_mass(
            mass,
            vel_mass
        )
        st.json({
            **seismic_mass_result,
            "description": f"The impact would generate seismic waves equivalent to a Magnitude {seismic_mass_result['magnitude']:.2f} earthquake."
        })

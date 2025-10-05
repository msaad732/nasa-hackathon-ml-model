# scripts/calibrate_usgs.py
import pandas as pd
import json
import os
from sklearn.linear_model import LinearRegression

IN_CSV = "data/usgs_earthquakes.csv"
OUT_JSON = "models/usgs_calibration.json"

def main():
    df = pd.read_csv(IN_CSV).dropna(subset=["magnitude"])
    mags = pd.to_numeric(df["magnitude"], errors="coerce").dropna()
    X = mags.values.reshape(-1,1)
    y = (1.5 * mags + 4.8).values.reshape(-1,1)  # theoretical log10(E)
    model = LinearRegression().fit(X, y)
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])

    os.makedirs("models", exist_ok=True)
    payload = {
        "slope": slope,
        "intercept": intercept,
        "coupling_factor": 1e-4  # fallback, can tune later
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved calibration → {OUT_JSON}")
    print(payload)

if __name__ == "__main__":
    main()


# # scripts/calibrate_usgs.py
# import pandas as pd
# import json
# import os
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score

# IN_CSV = "data/usgs_earthquakes.csv"
# OUT_JSON = "models/usgs_calibration.json"
# OUT_PLOT = "models/usgs_calibration_plot.png"

# def main():
#     # Load earthquake catalog
#     df = pd.read_csv(IN_CSV).dropna(subset=["magnitude"])
#     mags = pd.to_numeric(df["magnitude"], errors="coerce").dropna()

#     # Features and labels
#     X = mags.values.reshape(-1, 1)
#     y = (1.5 * mags + 4.8).values.reshape(-1, 1)  # theoretical log10(E)

#     # Fit regression
#     model = LinearRegression().fit(X, y)
#     y_pred = model.predict(X)

#     slope = float(model.coef_[0][0])
#     intercept = float(model.intercept_[0])
#     r2 = r2_score(y, y_pred)

#     # Save calibration constants
#     os.makedirs("models", exist_ok=True)
#     payload = {
#         "slope": slope,
#         "intercept": intercept,
#         "r2": r2,
#         "coupling_factor": 1e-4  # fallback, can tune later
#     }
#     with open(OUT_JSON, "w") as f:
#         json.dump(payload, f, indent=2)

#     # Make diagnostic plot
#     plt.figure(figsize=(6,4))
#     plt.scatter(X, y, s=10, alpha=0.3, label="USGS data (M vs log10(E))")
#     plt.plot(X, y_pred, color="red", linewidth=2, label="Fitted regression")
#     plt.xlabel("Magnitude (M)")
#     plt.ylabel("log10(E)")
#     plt.title("USGS Calibration Regression")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(OUT_PLOT)
#     plt.close()

#     print(f"Saved calibration → {OUT_JSON}")
#     print(f"Saved plot → {OUT_PLOT}")
#     print("Results:")
#     print(f"  slope = {slope:.3f}")
#     print(f"  intercept = {intercept:.3f}")
#     print(f"  R² = {r2:.4f}")

# if __name__ == "__main__":
#     main()

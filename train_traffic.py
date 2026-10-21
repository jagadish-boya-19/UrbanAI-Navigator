"""
UrbanAI Navigator - Multi-City Traffic Prediction Model Training
Trains an XGBoost Regressor on the calibrated 6-city dataset.
Evaluates R², MAE, RMSE, and classification accuracy.
Saves model artifacts and performance metrics for the Streamlit UI.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, accuracy_score

print("=" * 60)
print("  UrbanAI Navigator — 6-City Traffic Model Training Pipeline")
print("=" * 60)

# ── 1. Load Dataset ──────────────────────────────────────────────────
print("\n[1/5] Loading multi-city dataset...")
df = pd.read_csv("data/multi_city_traffic_dataset.csv")

print(f"      Total records  : {len(df):,}")
print(f"      Cities         : {df['City'].nunique()} ({', '.join(sorted(df['City'].unique()))})")
print(f"      Target metrics : Congestion_Percentage (continuous), Congestion_Level (discrete)")

# ── 2. Feature Engineering & Encodings ───────────────────────────────
print("\n[2/5] Engineering spatial and cyclical time features...")

# Encodings
CITY_MAP = {c: i for i, c in enumerate(sorted(df["City"].unique()))}
WEATHER_MAP = {"Clear": 0, "Overcast": 1, "Fog": 2, "Rain": 3, "Heavy Rain": 4}
CORRIDOR_MAP = {"Arterial Road": 0, "Tech Corridor": 1, "Transit / Highway": 2}
ROADWORK_MAP = {"No": 0, "Yes": 1}

df["City_Code"] = df["City"].map(CITY_MAP)
df["Weather_Code"] = df["Weather"].map(WEATHER_MAP).fillna(0).astype(int)
df["Corridor_Code"] = df["Corridor_Type"].map(CORRIDOR_MAP).fillna(0).astype(int)
df["Roadwork_Code"] = df["Roadwork"].map(ROADWORK_MAP).fillna(0).astype(int)

# Cyclical time encodings
df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour_Of_Day"] / 24.0)
df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour_Of_Day"] / 24.0)
df["Day_Sin"] = np.sin(2 * np.pi * df["Day_Of_Week"] / 7.0)
df["Day_Cos"] = np.cos(2 * np.pi * df["Day_Of_Week"] / 7.0)

# Holiday proximity normalization
df["Holiday_Proximity_Code"] = df["Holiday_Proximity"].replace({-1: 1, 0: 2, 1: 3, 999: 0})

FEATURES = [
    "City_Code",
    "Distance_km",
    "Corridor_Code",
    "Month",
    "Hour_Of_Day",
    "Hour_Sin",
    "Hour_Cos",
    "Day_Of_Week",
    "Day_Sin",
    "Day_Cos",
    "Is_Weekend",
    "Is_Peak_Hour",
    "Is_Holiday",
    "Holiday_Proximity_Code",
    "Weather_Code",
    "Roadwork_Code",
    "Incident_Reports",
    "Road_Capacity_Utilization",
]

X = df[FEATURES]
y = df["Congestion_Percentage"]

# ── 3. Train / Test Split ────────────────────────────────────────────
print("\n[3/5] Splitting dataset into train (80%) and test (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"      Training set : {len(X_train):,} samples")
print(f"      Testing set  : {len(X_test):,} samples")

# ── 4. Train XGBoost Model ───────────────────────────────────────────
print("\n[4/5] Training XGBoost Regressor (180 estimators, max_depth=6)...")

model = xgb.XGBRegressor(
    n_estimators=180,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ── 5. Evaluation & Validation ───────────────────────────────────────
print("\n[5/5] Evaluating model performance...")

preds = model.predict(X_test)
preds_clipped = np.clip(preds, 5.0, 98.0)

r2 = r2_score(y_test, preds_clipped)
mae = mean_absolute_error(y_test, preds_clipped)
rmse = root_mean_squared_error(y_test, preds_clipped)

# Map continuous predictions to discrete classes
def bin_congestion(scores):
    return ["High" if s >= 70.0 else "Medium" if s >= 35.0 else "Low" for s in scores]

y_test_classes = bin_congestion(y_test.values)
pred_classes = bin_congestion(preds_clipped)
acc = accuracy_score(y_test_classes, pred_classes)

print("-" * 50)
print(f"  R² Score (Variance Explained) : {r2 * 100:.2f}%")
print(f"  Mean Absolute Error (MAE)     : ±{mae:.2f}% congestion")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}%")
print(f"  Classification Accuracy       : {acc * 100:.2f}%")
print("-" * 50)

# Feature importances
importances = dict(zip(FEATURES, [round(float(v), 4) for v in model.feature_importances_]))
sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

print("\n  Top 5 Important Features:")
for feat, imp in list(sorted_importances.items())[:5]:
    print(f"    • {feat:<26}: {imp * 100:.1f}%")

# ── 6. Save Model Artifacts ──────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# 1. Save pickle model
with open("models/traffic_model.pkl", "wb") as f:
    pickle.dump(model, f)

# 2. Save json backup
model.save_model("models/traffic_model.json")

# 3. Save feature list
with open("models/traffic_features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

# 4. Save city & metadata mappings
mappings = {
    "CITY_MAP": CITY_MAP,
    "WEATHER_MAP": WEATHER_MAP,
    "CORRIDOR_MAP": CORRIDOR_MAP,
    "ROADWORK_MAP": ROADWORK_MAP,
    "FEATURES": FEATURES,
}
with open("models/traffic_city_mapping.pkl", "wb") as f:
    pickle.dump(mappings, f)

# 5. Save metrics for the UI viva helper
metrics_data = {
    "algorithm": "XGBoost Regressor",
    "n_estimators": 180,
    "max_depth": 6,
    "learning_rate": 0.08,
    "r2_score": round(float(r2 * 100), 2),
    "mae": round(float(mae), 2),
    "rmse": round(float(rmse), 2),
    "accuracy": round(float(acc * 100), 2),
    "feature_importances": sorted_importances,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "cities": list(CITY_MAP.keys()),
}
with open("models/traffic_metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=2)

print("\n" + "=" * 60)
print("  All artifacts saved successfully:")
print("  • models/traffic_model.pkl")
print("  • models/traffic_model.json")
print("  • models/traffic_features.pkl")
print("  • models/traffic_city_mapping.pkl")
print("  • models/traffic_metrics.json")
print("=" * 60)

# ── 7. Multi-City Live Test ──────────────────────────────────────────
print("\n  Quick Multi-City Live Verification:")

test_cases = [
    {"city": "Bangalore", "dist": 14.5, "corr": 1, "h": 9,  "day": 1, "rain": 0, "cap": 88, "name": "Bangalore: Silk Board -> Whitefield (Tue 9 AM)"},
    {"city": "Mumbai",    "dist": 11.2, "corr": 1, "h": 18, "day": 4, "rain": 3, "cap": 92, "name": "Mumbai: Bandra -> Andheri (Fri 6 PM Heavy Rain)"},
    {"city": "Delhi",     "dist": 16.0, "corr": 2, "h": 14, "day": 2, "rain": 0, "cap": 58, "name": "Delhi: Connaught Place -> Dwarka (Wed 2 PM)"},
    {"city": "Hyderabad", "dist": 8.5,  "corr": 1, "h": 10, "day": 0, "rain": 0, "cap": 82, "name": "Hyderabad: Hitech City -> Gachibowli (Mon 10 AM)"},
    {"city": "Chennai",   "dist": 12.0, "corr": 2, "h": 20, "day": 5, "rain": 0, "cap": 60, "name": "Chennai: T Nagar -> Velachery (Sat 8 PM)"},
    {"city": "Pune",      "dist": 18.0, "corr": 1, "h": 3,  "day": 6, "rain": 0, "cap": 25, "name": "Pune: Kothrud -> Hinjewadi (Sun 3 AM Late Night)"},
]

for tc in test_cases:
    city_code = CITY_MAP[tc["city"]]
    h = tc["h"]
    day = tc["day"]
    h_sin = np.sin(2 * np.pi * h / 24.0)
    h_cos = np.cos(2 * np.pi * h / 24.0)
    d_sin = np.sin(2 * np.pi * day / 7.0)
    d_cos = np.cos(2 * np.pi * day / 7.0)
    is_weekend = 1 if day >= 5 else 0
    is_peak = 1 if (8 <= h <= 11 or 17 <= h <= 21) else 0

    inp_df = pd.DataFrame([{
        "City_Code": city_code,
        "Distance_km": tc["dist"],
        "Corridor_Code": tc["corr"],
        "Month": 6,
        "Hour_Of_Day": h,
        "Hour_Sin": h_sin,
        "Hour_Cos": h_cos,
        "Day_Of_Week": day,
        "Day_Sin": d_sin,
        "Day_Cos": d_cos,
        "Is_Weekend": is_weekend,
        "Is_Peak_Hour": is_peak,
        "Is_Holiday": 0,
        "Holiday_Proximity_Code": 0,
        "Weather_Code": tc["rain"],
        "Roadwork_Code": 0,
        "Incident_Reports": 0,
        "Road_Capacity_Utilization": tc["cap"],
    }])[FEATURES]

    pred_score = float(np.clip(model.predict(inp_df)[0], 5.0, 98.0))
    lvl = "High" if pred_score >= 70.0 else "Medium" if pred_score >= 35.0 else "Low"
    badge = "[HIGH]" if lvl == "High" else "[MED] " if lvl == "Medium" else "[LOW] "
    print(f"  {badge} {tc['name']:<55} -> {lvl:<6} ({pred_score:.1f}% congestion)")

print("\n  Multi-City Verification Complete! Models ready for Streamlit deployment.")

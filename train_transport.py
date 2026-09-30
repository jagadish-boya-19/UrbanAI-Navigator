import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os

print("=" * 55)
print("  UrbanAI Navigator — Transport Crowd Model Training")
print("=" * 55)

# ── STEP 1: Load dataset ─────────────────────────────────────
print("\n[1/5] Loading dataset...")

df = pd.read_csv("data/transport_crowd_dataset.csv")

print(
    f"      Rows: {len(df):,}  |  "
    f"Stations: {df['Station_Name'].nunique()}"
)

# ── STEP 2: Create crowd labels ──────────────────────────────
print("\n[2/5] Creating crowd level labels...")

df["Crowd_Level"] = pd.cut(
    df["Crowd_Density_Percent"],
    bins=[0, 40, 74, 100],
    labels=["Low", "Medium", "High"]
)

print("      Label distribution:")

counts = df["Crowd_Level"].value_counts()

for label, count in counts.items():
    pct = count / len(df) * 100
    print(
        f"        {label:<8}: "
        f"{count:>8,} rows ({pct:.1f}%)"
    )

# ── STEP 3: Encode features ──────────────────────────────────
print("\n[3/5] Encoding features...")

# Convert Station_ID into numerical category codes
df["Station_Code"] = (
    df["Station_ID"]
    .astype("category")
    .cat.codes
)

# Save station mapping
station_map = (
    df[
        [
            "City",
            "Station_Name",
            "Station_ID",
            "Station_Code",
            "Transit_Type"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        ["City", "Station_Name"]
    )
)

os.makedirs("data", exist_ok=True)

station_map.to_csv(
    "data/station_mapping.csv",
    index=False
)

print(
    "      Station mapping saved to "
    "data/station_mapping.csv"
)

# Features
FEATURES = [
    "Station_Code",
    "Month",
    "Hour_Of_Day",
    "Day_Of_Week",
    "Is_Weekend",
    "Is_Holiday",
    "Is_Raining",
]

# Input features
X = df[FEATURES]

# Target
y = df["Crowd_Density_Percent"]

# ── STEP 4: Train XGBoost ────────────────────────────────────
print("\n[4/5] Training XGBoost model...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(
    f"      Train: {len(X_train):,} rows  |  "
    f"Test: {len(X_test):,} rows"
)

model = xgb.XGBRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ── STEP 5: Evaluate and save ────────────────────────────────
print("\n[5/5] Evaluating and saving...")

preds = model.predict(X_test)

r2 = r2_score(
    y_test,
    preds
)

mae = mean_absolute_error(
    y_test,
    preds
)

print(
    f"\n      R² Score : {r2 * 100:.2f}%"
)

print(
    f"      MAE      : ±{mae:.2f}% crowd density"
)

# Create models folder
os.makedirs(
    "models",
    exist_ok=True
)

# ── SAVE XGBOOST MODEL AS JSON ───────────────────────────────
model.save_model(
    "models/transport_model.json"
)

# ── SAVE XGBOOST MODEL AS PICKLE ─────────────────────────────
# This is the important fix.
with open(
    "models/transport_model.pkl",
    "wb"
) as f:
    pickle.dump(
        model,
        f
    )

# ── SAVE FEATURE LIST ────────────────────────────────────────
with open(
    "models/transport_features.pkl",
    "wb"
) as f:
    pickle.dump(
        FEATURES,
        f
    )

print("\n" + "=" * 55)
print(
    "  Model saved  : "
    "models/transport_model.pkl"
)

print(
    "  Model backup : "
    "models/transport_model.json"
)

print(
    "  Features     : "
    "models/transport_features.pkl"
)

print(
    "  Station map  : "
    "data/station_mapping.csv"
)

print("=" * 55)

# ── QUICK PREDICTION TEST ────────────────────────────────────
print("\n  Quick test predictions:")

station_map_loaded = pd.read_csv(
    "data/station_mapping.csv"
)

test_cases = [
    {
        "Station_Name": "Silk Board",
        "City": "Bangalore",
        "Hour": 8,
        "Day": 0,
        "label": "Bangalore Silk Board - Mon 8AM"
    },
    {
        "Station_Name": "Majestic (KSR)",
        "City": "Bangalore",
        "Hour": 14,
        "Day": 2,
        "label": "Bangalore Majestic - Wed 2PM"
    },
    {
        "Station_Name": "Rajiv Chowk",
        "City": "Delhi",
        "Hour": 9,
        "Day": 4,
        "label": "Delhi Rajiv Chowk - Fri 9AM"
    },
    {
        "Station_Name": "Andheri",
        "City": "Mumbai",
        "Hour": 18,
        "Day": 0,
        "label": "Mumbai Andheri - Mon 6PM"
    }
]

for tc in test_cases:

    row = station_map_loaded[
        (
            station_map_loaded["Station_Name"]
            == tc["Station_Name"]
        )
        &
        (
            station_map_loaded["City"]
            == tc["City"]
        )
    ]

    if row.empty:
        print(
            f"  Station not found: "
            f"{tc['Station_Name']}"
        )
        continue

    station_code = (
        row["Station_Code"]
        .values[0]
    )

    inp = pd.DataFrame(
        [
            {
                "Station_Code": station_code,
                "Month": 6,
                "Hour_Of_Day": tc["Hour"],
                "Day_Of_Week": tc["Day"],
                "Is_Weekend": (
                    1
                    if tc["Day"] >= 5
                    else 0
                ),
                "Is_Holiday": 0,
                "Is_Raining": 0,
            }
        ]
    )[FEATURES]

    prediction = model.predict(inp)[0]

    crowd_pct = int(
        np.clip(
            prediction,
            0,
            100
        )
    )

    if crowd_pct >= 75:
        level = "High"
        emoji = "🔴"

    elif crowd_pct >= 40:
        level = "Medium"
        emoji = "🟠"

    else:
        level = "Low"
        emoji = "🟢"

    print(
        f"  {emoji} "
        f"{tc['label']:<45} "
        f"→ {level} ({crowd_pct}%)"
    )

print(
    "\n  Training complete! "
    "Ready to use in Streamlit app."
)

print(
    "\n  IMPORTANT: "
    "transport_model.pkl has been created."
)
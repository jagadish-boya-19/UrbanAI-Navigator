import pandas as pd
import numpy as np
import datetime

np.random.seed(42)

# 15 Indian transit stations across 5 cities
stations = [
    # Delhi
    {"id": "DEL-M01", "name": "Rajiv Chowk",      "city": "Delhi",     "type": "Metro"},
    {"id": "DEL-M02", "name": "Hauz Khas",         "city": "Delhi",     "type": "Metro"},
    {"id": "DEL-M03", "name": "Kashmere Gate",     "city": "Delhi",     "type": "Metro"},
    {"id": "DEL-M04", "name": "Noida Sector 62",   "city": "Delhi",     "type": "Metro"},
    {"id": "DEL-B01", "name": "Anand Vihar ISBT",  "city": "Delhi",     "type": "Bus"},

    # Bangalore
    {"id": "BLR-M01", "name": "Majestic (KSR)",    "city": "Bangalore", "type": "Metro"},
    {"id": "BLR-M02", "name": "Indiranagar",       "city": "Bangalore", "type": "Metro"},
    {"id": "BLR-B01", "name": "Silk Board",        "city": "Bangalore", "type": "Bus"},
    {"id": "BLR-B02", "name": "Marathahalli",      "city": "Bangalore", "type": "Bus"},
    {"id": "BLR-B03", "name": "Hebbal",            "city": "Bangalore", "type": "Bus"},

    # Mumbai
    {"id": "MUM-M01", "name": "Ghatkopar",         "city": "Mumbai",    "type": "Metro"},
    {"id": "MUM-M02", "name": "Andheri",           "city": "Mumbai",    "type": "Metro"},
    {"id": "MUM-M03", "name": "Dadar",             "city": "Mumbai",    "type": "Metro"},
    {"id": "MUM-B01", "name": "Bandra West",       "city": "Mumbai",    "type": "Bus"},
    {"id": "MUM-B02", "name": "Borivali Station",  "city": "Mumbai",    "type": "Bus"},

    # Hyderabad
    {"id": "HYD-M01", "name": "Hitech City",       "city": "Hyderabad", "type": "Metro"},
    {"id": "HYD-M02", "name": "Ameerpet",          "city": "Hyderabad", "type": "Metro"},
    {"id": "HYD-B01", "name": "Secunderabad",      "city": "Hyderabad", "type": "Bus"},

    # Chennai
    {"id": "CHN-M01", "name": "Koyambedu",         "city": "Chennai",   "type": "Bus"},
    {"id": "CHN-M02", "name": "T Nagar",           "city": "Chennai",   "type": "Bus"},
]

# Indian national holidays (month, day)
indian_holidays = [(1,26), (8,15), (10,2), (12,25), (1,1), (5,1)]

start_date       = datetime.datetime(2025, 1, 1)
hours_to_generate = 365 * 24

print(f"Generating dataset for {len(stations)} stations x {hours_to_generate} hours...")
print("This may take 30-60 seconds...")

data = []

for station in stations:
    for h in range(hours_to_generate):
        current_time = start_date + datetime.timedelta(hours=h)
        hour         = current_time.hour
        day_idx      = current_time.weekday()
        month        = current_time.month
        is_weekend   = 1 if day_idx >= 5 else 0
        is_holiday   = 1 if (month, current_time.day) in indian_holidays else 0

        # Crowd density formula
        if is_weekend:
            base = 20 + 40 * np.exp(-((hour - 17) ** 2) / 12)
        else:
            p1   = 72 * np.exp(-((hour - 9)  ** 2) / 3.5)
            p2   = 76 * np.exp(-((hour - 18) ** 2) / 3.5)
            base = 12 + max(p1, p2)

        # Monsoon effect
        if month in [6, 7, 8, 9]:
            is_raining = 1 if np.random.rand() < 0.25 else 0
            if is_raining:
                base *= 1.22
        else:
            is_raining = 1 if np.random.rand() < 0.04 else 0
            if is_raining:
                base *= 1.12

        # Holiday reduces crowd
        if is_holiday:
            base *= 0.35

        # Late night / early morning
        if hour < 5 or hour > 23:
            base = max(1, base * 0.03)

        # Station type effect
        if station["type"] == "Metro":
            base *= 1.15

        # Add realistic noise
        noise = np.random.normal(0, 1.8)
        crowd_percentage = int(np.clip(base + noise, 0, 100))

        data.append({
            "Station_ID": station["id"],
            "Station_Name": station["name"],
            "City": station["city"],
            "Transit_Type": station["type"],
            "Month": month,
            "Hour_Of_Day": hour,
            "Day_Of_Week": day_idx,
            "Is_Weekend": is_weekend,
            "Is_Holiday": is_holiday,
            "Is_Raining": is_raining,
            "Crowd_Density_Percent": crowd_percentage,
        })

df = pd.DataFrame(data)
df.to_csv("data/transport_crowd_dataset.csv", index=False)

print("\nDataset created successfully!")
print(f"Total rows    : {len(df):,}")
print(f"Cities covered: {df['City'].nunique()}")
print(f"Stations      : {df['Station_Name'].nunique()}")

print("\nCrowd level distribution:")
df["Crowd_Level"] = pd.cut(
    df["Crowd_Density_Percent"],
    bins=[0, 40, 74, 100],
    labels=["Low", "Medium", "High"]
)

print(df["Crowd_Level"].value_counts())
print("\nSaved to: data/transport_crowd_dataset.csv")
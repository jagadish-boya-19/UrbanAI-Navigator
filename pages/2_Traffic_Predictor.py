import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import date, timedelta
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Traffic Predictor - UrbanAI Navigator",
    layout="wide",
    page_icon="🚦"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1rem; }
    .stMetric {
        background: white; padding: 12px; border-radius: 10px;
        border-left: 4px solid #1F4E79;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stButton > button {
        background-color: #1F4E79; color: white;
        border-radius: 8px; border: none;
        font-weight: bold; width: 100%;
    }
    .stButton > button:hover { background-color: #2E86AB; }
    .result-High {
        background:#FDEDEC; border-left:5px solid #E74C3C;
        padding:16px 20px; border-radius:8px; margin:8px 0;
    }
    .result-Medium {
        background:#FEF9E7; border-left:5px solid #F39C12;
        padding:16px 20px; border-radius:8px; margin:8px 0;
    }
    .result-Low {
        background:#EAFAF1; border-left:5px solid #27AE60;
        padding:16px 20px; border-radius:8px; margin:8px 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ── City Data ────────────────────────────────────────────────
CITY_DATA = {
    "Bangalore": {
        "areas": {
            "MG Road":[12.9758,77.6085],"Silk Board":[12.9172,77.6231],
            "Whitefield":[12.9698,77.7500],"Koramangala":[12.9352,77.6245],
            "Hebbal":[13.0350,77.5970],"Electronic City":[12.8458,77.6603],
            "Jayanagar":[12.9308,77.5830],"Marathahalli":[12.9591,77.6971],
            "Indiranagar":[12.9784,77.6408],"Rajajinagar":[12.9918,77.5538],
            "Banashankari":[12.9255,77.5468],"JP Nagar":[12.9102,77.5847],
            "Yelahanka":[13.1007,77.5963],"Basavanagudi":[12.9428,77.5760],
            "Bellandur":[12.9261,77.6767],"Madiwala":[12.9214,77.6199],
            "Shivajinagar":[12.9858,77.6009],
        },
        "center":[12.9716,77.5946],"has_model":True,
    },
    "Mumbai": {
        "areas": {
            "Bandra":[19.0596,72.8295],"Andheri":[19.1136,72.8697],
            "Dadar":[19.0178,72.8478],"Powai":[19.1197,72.9051],
            "Thane":[19.2183,72.9781],"Worli":[19.0176,72.8152],
            "Borivali":[19.2307,72.8567],"Navi Mumbai":[19.0330,73.0297],
            "Juhu":[19.0883,72.8267],"Kurla":[19.0726,72.8797],
            "Malad":[19.1872,72.8486],"Goregaon":[19.1663,72.8497],
        },
        "center":[19.0760,72.8777],"has_model":False,
    },
    "Delhi": {
        "areas": {
            "Connaught Place":[28.6315,77.2167],"Lajpat Nagar":[28.5700,77.2433],
            "Dwarka":[28.5921,77.0460],"Rohini":[28.7041,77.1025],
            "Karol Bagh":[28.6519,77.1907],"Hauz Khas":[28.5494,77.2001],
            "Saket":[28.5244,77.2167],"Noida Sector 18":[28.5672,77.3211],
            "Vasant Kunj":[28.5200,77.1500],"Pitampura":[28.7016,77.1394],
            "Nehru Place":[28.5491,77.2519],"South Extension":[28.5706,77.2168],
        },
        "center":[28.6139,77.2090],"has_model":False,
    },
    "Hyderabad": {
        "areas": {
            "Hitech City":[17.4435,78.3772],"Charminar":[17.3616,78.4747],
            "Banjara Hills":[17.4156,78.4347],"Gachibowli":[17.4401,78.3489],
            "Secunderabad":[17.4399,78.4983],"Ameerpet":[17.4375,78.4482],
            "LB Nagar":[17.3493,78.5510],"Kukatpally":[17.4849,78.3996],
            "Begumpet":[17.4432,78.4672],"Jubilee Hills":[17.4316,78.4077],
            "Madhapur":[17.4487,78.3908],
        },
        "center":[17.3850,78.4867],"has_model":False,
    },
    "Chennai": {
        "areas": {
            "T Nagar":[13.0418,80.2341],"Anna Nagar":[13.0850,80.2101],
            "Adyar":[13.0012,80.2565],"Velachery":[12.9815,80.2180],
            "Tambaram":[12.9249,80.1000],"Ambattur":[13.1143,80.1548],
            "Porur":[13.0389,80.1574],"Egmore":[13.0732,80.2609],
            "Chrompet":[12.9516,80.1462],"Guindy":[13.0068,80.2206],
            "Perambur":[13.1152,80.2358],
        },
        "center":[13.0827,80.2707],"has_model":False,
    },
    "Pune": {
        "areas": {
            "Koregaon Park":[18.5362,73.8938],"Hinjewadi":[18.5912,73.7389],
            "Kothrud":[18.5074,73.8077],"Viman Nagar":[18.5679,73.9143],
            "Hadapsar":[18.5018,73.9252],"Baner":[18.5590,73.7868],
            "Katraj":[18.4529,73.8567],"Wakad":[18.5975,73.7618],
            "Shivajinagar":[18.5308,73.8474],"Aundh":[18.5588,73.8087],
            "Magarpatta":[18.5131,73.9282],
        },
        "center":[18.5204,73.8567],"has_model":False,
    },
}

WEATHER_MAP = {"Clear":0,"Overcast":1,"Windy":2,"Fog":3,"Rain":4}
DAY_MAP     = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
               "Friday":4,"Saturday":5,"Sunday":6}
COLOR_MAP   = {"High":"#E74C3C","Medium":"#F39C12","Low":"#27AE60"}
FOLIUM_COLOR= {"High":"red","Medium":"orange","Low":"green"}
EMOJI_MAP   = {"High":"🔴","Medium":"🟠","Low":"🟢"}

# ── Load ML model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained Bangalore model and its metadata safely.

    The app falls back to the rule engine when the model files are not
    available, but unexpected errors are reported instead of being silently
    swallowed.
    """
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models"
    try:
        with open(model_dir / "traffic_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)
        with open(model_dir / "traffic_features.pkl", "rb") as f:
            loaded_features = pickle.load(f)
        with open(model_dir / "area_mapping.pkl", "rb") as f:
            loaded_area_mapping = pickle.load(f)
        return loaded_model, list(loaded_features), loaded_area_mapping
    except FileNotFoundError:
        # Missing model files are an expected fallback condition.
        st.warning(
            "ML model files not found. Rule-based engine will be used."
        )
        return None, None, None
    except Exception as e:
        st.error(f"Could not load the ML model files: {type(e).__name__}: {e}")
        return None, None, None

model, features, area_mapping = load_model()

# ── Forecast profiles ────────────────────────────────────────
# Deterministic baseline factors used to create simulated forecast inputs.
# They are not claimed to be live traffic measurements.
HOURLY_TRAFFIC_FACTOR = {
    0:0.40, 1:0.35, 2:0.30, 3:0.30, 4:0.35, 5:0.45,
    6:0.65, 7:1.15, 8:1.35, 9:1.30, 10:1.10, 11:0.95,
    12:0.90, 13:0.90, 14:0.92, 15:0.98, 16:1.10, 17:1.30,
    18:1.45, 19:1.40, 20:1.20, 21:0.90, 22:0.70, 23:0.50,
}
DAY_TRAFFIC_FACTOR = {
    0:1.05, 1:1.10, 2:1.10, 3:1.12, 4:1.20, 5:0.85, 6:0.65
}

def clamp(value, low, high):
    return max(low, min(high, value))

# ── Rule-based engine ────────────────────────────────────────
def rule_predict(hour, day_num, weather, roadwork_str,
                 traffic_volume, avg_speed, capacity_util):
    score = 0
    if 7 <= hour <= 10 or 17 <= hour <= 20: score += 40
    elif 10 < hour < 17:                     score += 15
    elif 20 < hour <= 22:                    score += 10
    if day_num < 5:  score += 20
    else:            score -= 10
    weather_scores = {"Clear":0,"Overcast":5,"Windy":5,"Fog":20,"Rain":25}
    score += weather_scores.get(weather, 0)
    if roadwork_str == "Yes": score += 20
    if traffic_volume > 3500:  score += 20
    elif traffic_volume > 2000: score += 10
    if avg_speed < 20:    score += 20
    elif avg_speed < 35:  score += 10
    if capacity_util > 85: score += 15
    elif capacity_util > 60: score += 5
    score = clamp(score, 0, 100)
    pred = "High" if score >= 70 else "Medium" if score >= 35 else "Low"
    return pred, score

# ── Prediction helpers ───────────────────────────────────────
def _travel_time_index(speed):
    """Return the existing project's TTI formula.

    IMPORTANT: This must match the definition used when the trained model was
    created. If the training dataset uses a different definition, replace this
    formula with the dataset's exact definition rather than changing the model.
    """
    return round(1 + (100 - speed) / 50, 2)

def predict(city, model_area, hour, selected_date, weather, roadwork,
            tv, speed, cap, pub, ped, sig, inc):
    day_num = selected_date.weekday()
    month_num = selected_date.month
    is_weekend = 1 if day_num >= 5 else 0
    is_peak = 1 if (7 <= hour <= 10 or 17 <= hour <= 20) else 0

    if city == "Bangalore" and model is not None:
        if not isinstance(area_mapping, dict) or model_area not in area_mapping:
            raise ValueError(
                f"Model area '{model_area}' is not present in the trained area mapping. "
                "The model cannot safely encode this area."
            )
        area_code = area_mapping[model_area]
        row = {
            "hour": hour, "day_of_week": day_num, "month": month_num,
            "is_weekend": is_weekend, "is_peak": is_peak,
            "Traffic Volume": tv, "Average Speed": speed,
            "Travel Time Index": _travel_time_index(speed),
            "Road Capacity Utilization": cap, "Incident Reports": inc,
            "Public Transport Usage": pub,
            "Traffic Signal Compliance": sig,
            "Pedestrian and Cyclist Count": ped,
            "weather_encoded": WEATHER_MAP[weather],
            "roadwork_encoded": 1 if roadwork == "Yes" else 0,
            "area_encoded": area_code,
        }
        missing = [f for f in features if f not in row]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        df_in = pd.DataFrame([row])[features]
        pred = model.predict(df_in)[0]
        if hasattr(model, "predict_proba"):
            raw_proba = model.predict_proba(df_in)[0]
            proba = dict(zip(model.classes_, raw_proba))
        else:
            proba = {}
        return pred, proba, True, None

    pred, score = rule_predict(
        hour, day_num, weather, roadwork, tv, speed, cap
    )
    return pred, {}, False, score

def generate_forecast_inputs(hour, base_tv, base_speed, base_cap,
                             base_pub, base_ped, base_sig, base_inc,
                             day_factor=1.0):
    """Generate deterministic, bounded simulated traffic conditions."""
    factor = HOURLY_TRAFFIC_FACTOR[hour] * day_factor
    tv_value = clamp(base_tv * factor, 100, 5000)

    # Higher volume generally corresponds to lower speed in this simulation.
    speed_ratio = clamp(1.0 - 0.30 * (factor - 0.50), 0.55, 1.15)
    speed_value = clamp(base_speed * speed_ratio, 5, 80)

    cap_value = clamp(base_cap * (0.90 + 0.10 * factor), 10, 100)
    pub_value = clamp(base_pub * (0.85 + 0.15 * factor), 0, 100)
    ped_value = clamp(base_ped * (0.55 + 0.45 * factor), 0, 500)
    sig_value = clamp(base_sig - max(0, factor - 1.0) * 8, 40, 100)
    inc_value = clamp(round(base_inc * (0.60 + 0.40 * factor)), 0, 10)

    return {
        "tv": round(tv_value), "speed": round(speed_value, 1),
        "cap": round(cap_value, 1), "pub": round(pub_value, 1),
        "ped": round(ped_value), "sig": round(sig_value, 1),
        "inc": inc_value,
    }

def hourly_forecast(city, model_area, selected_date, weather, roadwork,
                    tv, speed, cap, pub, ped, sig, inc):
    results = []
    for h in range(24):
        vals = generate_forecast_inputs(
            h, tv, speed, cap, pub, ped, sig, inc,
            DAY_TRAFFIC_FACTOR[selected_date.weekday()]
        )
        pred, proba, used_ml, score = predict(
            city, model_area, h, selected_date, weather, roadwork,
            vals["tv"], vals["speed"], vals["cap"], vals["pub"],
            vals["ped"], vals["sig"], vals["inc"]
        )
        results.append({
            "hour": h, "pred": pred, "proba": proba,
            "used_ml": used_ml, "score": score, **vals
        })
    return results

def weekly_forecast(city, model_area, hour, selected_date, weather, roadwork,
                    tv, speed, cap, pub, ped, sig, inc):
    results = []
    for offset in range(7):
        forecast_date = selected_date + timedelta(days=offset)
        day_factor = DAY_TRAFFIC_FACTOR[forecast_date.weekday()]
        vals = generate_forecast_inputs(
            hour, tv, speed, cap, pub, ped, sig, inc, day_factor
        )
        pred, proba, used_ml, score = predict(
            city, model_area, hour, forecast_date, weather, roadwork,
            vals["tv"], vals["speed"], vals["cap"], vals["pub"],
            vals["ped"], vals["sig"], vals["inc"]
        )
        results.append({
            "date": forecast_date,
            "day": forecast_date.strftime("%a %d"),
            "full_day": forecast_date.strftime("%A"),
            "pred": pred, "proba": proba,
            "used_ml": used_ml, "score": score, **vals
        })
    return results

# ── Charts ───────────────────────────────────────────────────
def hourly_chart(preds, sel_hour):
    nums   = [3 if p=="High" else 2 if p=="Medium" else 1 for p in preds]
    colors = [COLOR_MAP[p] for p in preds]
    fig = go.Figure(go.Bar(
        x=list(range(24)), y=nums, marker_color=colors,
        text=[p[0] for p in preds], textposition="outside",
        hovertemplate="Hour %{x}:00 — %{text}<extra></extra>",
    ))
    fig.add_vline(x=sel_hour, line_dash="dash", line_color="#1F4E79",
                  line_width=2,
                  annotation_text=f"Selected ({sel_hour}:00)",
                  annotation_position="top right",
                  annotation_font_size=10)
    fig.update_layout(
        height=230, margin=dict(t=30,b=30,l=10,r=10),
        xaxis=dict(title="Hour of Day", dtick=2, tickfont_size=10),
        yaxis=dict(tickvals=[1,2,3],ticktext=["Low","Medium","High"],title=""),
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

def weekly_chart(weekly):
    labels = [w["day"] for w in weekly]
    nums = [3 if w["pred"] == "High" else 2 if w["pred"] == "Medium" else 1 for w in weekly]
    colors = [COLOR_MAP[w["pred"]] for w in weekly]
    preds = [w["pred"] for w in weekly]
    fig = go.Figure(go.Bar(
        x=labels, y=nums, marker_color=colors,
        text=preds, textposition="outside",
        hovertemplate="%{x} — %{text}<extra></extra>",
    ))
    fig.update_layout(
        height=210, margin=dict(t=20,b=20,l=10,r=10),
        xaxis=dict(title="Date", tickfont_size=10),
        yaxis=dict(tickvals=[1,2,3], ticktext=["Low","Medium","High"], title=""),
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

def probability_chart(proba):
    labels = list(proba.keys())
    values = [round(float(v) * 100) for v in proba.values()]
    colors = [COLOR_MAP.get(l, "#888") for l in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v}%" for v in values], textposition="outside",
    ))
    fig.update_layout(
        height=160, margin=dict(t=10,b=10,l=10,r=50),
        xaxis=dict(range=[0,115], showticklabels=False),
        yaxis=dict(title=""), showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

def score_chart(score):
    fig = go.Figure(go.Bar(
        x=[score], y=["Congestion Score"], orientation="h",
        marker_color=["#1F4E79"], text=[f"{score}/100"],
        textposition="outside",
    ))
    fig.update_layout(
        height=160, margin=dict(t=10,b=10,l=10,r=50),
        xaxis=dict(range=[0,115], showticklabels=False),
        yaxis=dict(title=""), showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

# ── Route / Model Location Helpers ──────────────────────────
def _nearest_area(city, coords, allowed=None):
    """Return the named area nearest to the supplied coordinates."""
    areas = CITY_DATA[city]["areas"]
    candidates = list(allowed) if allowed else list(areas.keys())
    candidates = [a for a in candidates if a in areas]
    if not candidates:
        return next(iter(areas.keys()))
    lat, lon = coords
    return min(candidates, key=lambda a: (areas[a][0] - lat) ** 2 + (areas[a][1] - lon) ** 2)


def get_model_area(city, source, destination):
    """Map the selected route to an internal area for the existing area-based ML model."""
    source_coords = CITY_DATA[city]["areas"][source]
    destination_coords = CITY_DATA[city]["areas"][destination]
    midpoint = ((source_coords[0] + destination_coords[0]) / 2,
                (source_coords[1] + destination_coords[1]) / 2)

    if city == "Bangalore" and isinstance(area_mapping, dict) and area_mapping:
        trained_areas = [name for name in area_mapping if name in CITY_DATA[city]["areas"]]
        if trained_areas:
            return _nearest_area(city, midpoint, trained_areas)
    return _nearest_area(city, midpoint)


def route_map(city, source, destination, pred):
    """Display source/destination markers and a visual connection between them."""
    source_coords = CITY_DATA[city]["areas"][source]
    destination_coords = CITY_DATA[city]["areas"][destination]
    midpoint = ((source_coords[0] + destination_coords[0]) / 2,
                (source_coords[1] + destination_coords[1]) / 2)
    m = folium.Map(location=midpoint, zoom_start=13, tiles="OpenStreetMap")

    folium.Marker(source_coords,
        popup=folium.Popup(f"<b>Source</b><br>{source}", max_width=200),
        tooltip=f"Source: {source}",
        icon=folium.Icon(color="green", icon="play", prefix="glyphicon")).add_to(m)
    folium.Marker(destination_coords,
        popup=folium.Popup(f"<b>Destination</b><br>{destination}", max_width=200),
        tooltip=f"Destination: {destination}",
        icon=folium.Icon(color="red", icon="flag", prefix="glyphicon")).add_to(m)
    folium.PolyLine([source_coords, destination_coords],
        color=COLOR_MAP[pred], weight=6, opacity=0.75, dash_array="8, 8",
        tooltip=f"Predicted congestion: {pred}").add_to(m)
    return m

# ================================================================
# UI LAYOUT
# ================================================================
st.title("🚦 Traffic Congestion Predictor")
st.markdown("*Current prediction + 24-hour forecast + weekly forecast — all Indian cities*")
st.markdown("---")

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.title("UrbanAI Navigator")
    st.markdown("*Smart City Life Optimizer*")
    st.markdown("---")
    st.markdown("### Prediction Settings")

    city = st.selectbox("Select City", list(CITY_DATA.keys()))
    location_options = sorted(CITY_DATA[city]["areas"].keys())
    source = st.selectbox("Source", location_options, index=0)
    destination_options = [loc for loc in location_options if loc != source]
    if not destination_options:
        destination_options = location_options
    destination = st.selectbox("Destination", destination_options, index=0)

    st.markdown("---")
    selected_date = st.date_input(
        "Select Date", value=date.today(),
        min_value=date(2020, 1, 1), max_value=date(2035, 12, 31),
    )
    hour = st.slider("Time", 0, 23, 8, format="%d:00")
    st.caption(f"Selected time: **{hour:02d}:00**")

    # Kept internally for compatibility with the trained model; no longer user controls.
    weather = "Clear"
    roadwork = "No"
    tv, speed, cap, pub, ped, sig, inc = 2500, 28, 70, 45, 120, 75, 1

    model_area = get_model_area(city, source, destination)

    st.markdown("---")
    if "predicted" not in st.session_state:
        st.session_state.predicted = False
    if st.button("Predict Congestion", type="primary", use_container_width=True):
        st.session_state.predicted = True

    st.markdown("---")
    if city == "Bangalore" and model is not None:
        st.success("ML Model Active\nRandom Forest ML")
    elif city == "Bangalore":
        st.warning("ML model unavailable\nUsing Rule-based Simulation")
    else:
        st.info("Rule-based Simulation\nNo ML model for this city")

# ── TOP METRICS ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("City", city)
c2.metric("Source", source)
c3.metric("Destination", destination)
c4.metric("Date & Time", selected_date.strftime("%d %b %Y") + f" | {hour:02d}:00")
st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📍 Current Prediction",
    "📈 24-Hour Forecast",
    "📅 Weekly Forecast",
])

# ────────────────────────────────────────────────────────────
# TAB 1
# ────────────────────────────────────────────────────────────
with tab1:
    if not st.session_state.predicted:
        st.info("Select a source, destination, date and time in the sidebar, then click **Predict Congestion**.")
        st.subheader(f"{city} — Select a source and destination")
        m_def = folium.Map(location=CITY_DATA[city]["center"], zoom_start=12, tiles="OpenStreetMap")
        for a_name, coords in CITY_DATA[city]["areas"].items():
            folium.Marker(
                coords, tooltip=a_name, popup=a_name,
                icon=folium.Icon(color="blue", icon="map-marker", prefix="glyphicon")
            ).add_to(m_def)
        st_folium(m_def, width=1100, height=430, returned_objects=[])
    else:
        try:
            pred, proba, used_ml, score = predict(
                city, model_area, hour, selected_date, weather, roadwork,
                tv, speed, cap, pub, ped, sig, inc
            )
        except Exception as e:
            st.error(f"Prediction could not be generated: {e}")
            st.stop()

        c1.metric("City", city)
        c2.metric("Source", source)
        c3.metric("Congestion", pred, delta="Peak" if (7 <= hour <= 10 or 17 <= hour <= 20) else "Off-Peak")
        if used_ml and proba:
            c4.metric("Prediction Probability", f"{round(proba.get(pred, 0) * 100)}%")
        else:
            c4.metric("Congestion Score", f"{score}/100")

        left, middle, right = st.columns([1.2, 1.2, 1])

        with left:
            st.markdown(f"""
            <div class="result-{pred}">
                <h2 style="margin:0;color:{COLOR_MAP[pred]}">
                    {EMOJI_MAP[pred]} {pred} Congestion
                </h2>
                <p style="margin:6px 0 0;font-size:13px;color:#555">
                    {source} → {destination}, {city} &nbsp;|&nbsp; {selected_date.strftime('%d %B %Y')} {hour:02d}:00
                </p>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if pred == "High":
                st.error("**Consider avoiding travel if possible**")
                st.warning("Consider public transport or an alternate route")
            elif pred == "Medium":
                st.warning("**Expect some delays**")
                st.info("Allow some extra travel time")
            else:
                st.success("**Low congestion expected**")
                st.info("Conditions are relatively favorable based on the prediction")

            engine_text = "Random Forest ML" if used_ml else "Rule-based Simulation"
            st.markdown(f"<small style='color:#888'>Engine: <b>{engine_text}</b></small>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Best Travel Times Today**")
            with st.spinner("Checking today's hourly predictions..."):
                today_forecast = hourly_forecast(
                    city, model_area, selected_date, weather, roadwork,
                    tv, speed, cap, pub, ped, sig, inc
                )
            low_hours = [r["hour"] for r in today_forecast if r["pred"] == "Low"]
            high_hours = [r["hour"] for r in today_forecast if r["pred"] == "High"]
            if low_hours:
                st.success("Low-congestion hours: " + ", ".join(f"{h:02d}:00" for h in low_hours))
            else:
                st.info("No low-congestion hours predicted for this date.")
            if high_hours:
                st.warning("High-congestion hours: " + ", ".join(f"{h:02d}:00" for h in high_hours))

        with middle:
            if used_ml and proba:
                st.markdown("**Prediction Probability**")
                st.plotly_chart(probability_chart(proba), use_container_width=True)
            else:
                st.markdown("**Congestion Score**")
                st.plotly_chart(score_chart(score), use_container_width=True)
            st.markdown("**Traffic Metrics**")
            r1, r2 = st.columns(2)
            r1.metric("Speed", f"{speed} km/h")
            r2.metric("Volume", f"{tv}/hr")
            r3, r4 = st.columns(2)
            r3.metric("Capacity", f"{cap}%")
            r4.metric("Incidents", inc)

        with right:
            st.markdown(f"**{source} → {destination}**")
            st_folium(route_map(city, source, destination, pred), width=310, height=360, returned_objects=[])
            st.caption(f"Model area used internally: **{model_area}**")

# ────────────────────────────────────────────────────────────
# TAB 2
# ────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"24-Hour Traffic Forecast — {source} → {destination}, {city}")
    st.caption(f"{selected_date.strftime('%d %B %Y')} | Route-based view")

    with st.spinner("Generating 24-hour forecast..."):
        forecast_24 = hourly_forecast(
            city, model_area, selected_date, weather, roadwork,
            tv, speed, cap, pub, ped, sig, inc
        )

    preds_24 = [r["pred"] for r in forecast_24]
    high_h = [r["hour"] for r in forecast_24 if r["pred"] == "High"]
    medium_h = [r["hour"] for r in forecast_24 if r["pred"] == "Medium"]
    low_h = [r["hour"] for r in forecast_24 if r["pred"] == "Low"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High Hours", len(high_h), delta=f"Starts {high_h[0]:02d}:00" if high_h else "None")
    c2.metric("Medium Hours", len(medium_h))
    c3.metric("Low Congestion Hours", len(low_h))
    c4.metric("Earliest Low Hour", f"{low_h[0]:02d}:00" if low_h else "None")

    st.markdown("---")
    st.plotly_chart(hourly_chart(preds_24, hour), use_container_width=True)

    st.markdown("**Hour-by-Hour Breakdown**")
    table = []
    for r in forecast_24:
        i, p = r["hour"], r["pred"]
        period = ("Early Morning" if i < 6
                  else "Morning Peak" if 6 <= i <= 10
                  else "Midday" if 10 < i <= 16
                  else "Evening Peak" if 16 < i <= 20
                  else "Night")
        table.append({
            "Time": f"{i:02d}:00",
            "Period": period,
            "Congestion": f"{EMOJI_MAP[p]} {p}",
            "Traffic Volume": f"{r['tv']} /hr",
            "Avg Speed": f"{r['speed']} km/h",
            "Action": ("Travel now" if p == "Low" else "Allow extra time" if p == "Medium" else "Consider avoiding if possible"),
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True, height=360)

# ────────────────────────────────────────────────────────────
# TAB 3
# ────────────────────────────────────────────────────────────
with tab3:
    end_date = selected_date + timedelta(days=6)
    st.subheader(f"7-Day Traffic Forecast — {source} → {destination}, {city}")
    st.caption(f"{selected_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')} | At {hour:02d}:00")

    with st.spinner("Generating 7-day forecast..."):
        weekly = weekly_forecast(
            city, model_area, hour, selected_date, weather, roadwork,
            tv, speed, cap, pub, ped, sig, inc
        )

    st.plotly_chart(weekly_chart(weekly), use_container_width=True)

    cols = st.columns(7)
    for col, w in zip(cols, weekly):
        with col:
            st.markdown(f"""
            <div style="text-align:center;background:white;border-radius:8px;
                        padding:12px 4px;box-shadow:0 1px 4px rgba(0,0,0,0.08)">
                <div style="font-size:11px;color:#888;margin-bottom:4px">{w['day']}</div>
                <div style="font-size:10px;color:#777">{w['full_day']}</div>
                <div style="font-size:22px">{EMOJI_MAP[w['pred']]}</div>
                <div style="font-size:10px;font-weight:bold;
                            color:{COLOR_MAP[w['pred']]};margin-top:4px">
                    {w['pred']}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    low_days = [w["day"] for w in weekly if w["pred"] == "Low"]
    high_days = [w["day"] for w in weekly if w["pred"] == "High"]
    col1, col2 = st.columns(2)
    with col1:
        if low_days:
            st.success(f"**Low-congestion dates:** {', '.join(low_days)}")
        else:
            st.info("No low-congestion dates predicted in this 7-day window.")
    with col2:
        if high_days:
            st.warning(f"**High-congestion dates:** {', '.join(high_days)}")
        else:
            st.success("No high-congestion dates predicted in this 7-day window.")

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#888'>UrbanAI Navigator | Traffic Predictor | "
    "VTU Final Year Project 2025-26 | GEC Karwar, Dept of CSE</small></center>",
    unsafe_allow_html=True
)

import streamlit as st
import requests
import folium
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import datetime

load_dotenv()
OWM_KEY = os.getenv("OPENWEATHER_API_KEY")

st.set_page_config(
    page_title="Health Index - UrbanAI Navigator",
    layout="wide",
    page_icon="🌫️"
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
    .alert-danger {
        background:#FDEDEC; border-left:5px solid #E74C3C;
        padding:12px 16px; border-radius:8px; margin:6px 0;
    }
    .alert-warning {
        background:#FEF9E7; border-left:5px solid #F39C12;
        padding:12px 16px; border-radius:8px; margin:6px 0;
    }
    .alert-success {
        background:#EAFAF1; border-left:5px solid #27AE60;
        padding:12px 16px; border-radius:8px; margin:6px 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ── City coordinates ──────────────────────────────────────────
CITIES = {
    "Bangalore": {
        "center": [12.9716, 77.5946],
        "zones": {
            "MG Road":         {"coords":[12.9758,77.6085], "weight":0.9},
            "Silk Board":      {"coords":[12.9172,77.6231], "weight":0.95},
            "Hebbal":          {"coords":[13.0350,77.5970], "weight":0.8},
            "Marathahalli":    {"coords":[12.9591,77.6971], "weight":0.85},
            "Electronic City": {"coords":[12.8458,77.6603], "weight":0.75},
            "Whitefield":      {"coords":[12.9698,77.7500], "weight":0.55},
            "Koramangala":     {"coords":[12.9352,77.6245], "weight":0.60},
            "Jayanagar":       {"coords":[12.9308,77.5830], "weight":0.25},
            "Indiranagar":     {"coords":[12.9784,77.6408], "weight":0.50},
            "Rajajinagar":     {"coords":[12.9918,77.5538], "weight":0.20},
        }
    },
    "Mumbai": {
        "center": [19.0760, 72.8777],
        "zones": {
            "Bandra":    {"coords":[19.0596,72.8295], "weight":0.80},
            "Andheri":   {"coords":[19.1136,72.8697], "weight":0.85},
            "Dadar":     {"coords":[19.0178,72.8478], "weight":0.90},
            "Worli":     {"coords":[19.0176,72.8152], "weight":0.65},
            "Borivali":  {"coords":[19.2307,72.8567], "weight":0.35},
            "Powai":     {"coords":[19.1197,72.9051], "weight":0.55},
            "Thane":     {"coords":[19.2183,72.9781], "weight":0.50},
        }
    },
    "Delhi": {
        "center": [28.6139, 77.2090],
        "zones": {
            "Connaught Place": {"coords":[28.6315,77.2167], "weight":0.95},
            "Lajpat Nagar":    {"coords":[28.5700,77.2433], "weight":0.85},
            "Karol Bagh":      {"coords":[28.6519,77.1907], "weight":0.90},
            "Rohini":          {"coords":[28.7041,77.1025], "weight":0.55},
            "Hauz Khas":       {"coords":[28.5494,77.2001], "weight":0.60},
            "Vasant Kunj":     {"coords":[28.5200,77.1500], "weight":0.30},
            "Dwarka":          {"coords":[28.5921,77.0460], "weight":0.50},
        }
    },
    "Hyderabad": {
        "center": [17.3850, 78.4867],
        "zones": {
            "Hitech City":   {"coords":[17.4435,78.3772], "weight":0.75},
            "Charminar":     {"coords":[17.3616,78.4747], "weight":0.85},
            "Banjara Hills": {"coords":[17.4156,78.4347], "weight":0.65},
            "Gachibowli":    {"coords":[17.4401,78.3489], "weight":0.70},
            "LB Nagar":      {"coords":[17.3493,78.5510], "weight":0.30},
            "Kukatpally":    {"coords":[17.4849,78.3996], "weight":0.35},
        }
    },
    "Chennai": {
        "center": [13.0827, 80.2707],
        "zones": {
            "T Nagar":    {"coords":[13.0418,80.2341], "weight":0.80},
            "Anna Nagar": {"coords":[13.0850,80.2101], "weight":0.60},
            "Adyar":      {"coords":[13.0012,80.2565], "weight":0.55},
            "Velachery":  {"coords":[12.9815,80.2180], "weight":0.70},
            "Egmore":     {"coords":[13.0732,80.2609], "weight":0.75},
            "Tambaram":   {"coords":[12.9249,80.1000], "weight":0.40},
        }
    },
    "Pune": {
        "center": [18.5204, 73.8567],
        "zones": {
            "Koregaon Park": {"coords":[18.5362,73.8938], "weight":0.70},
            "Hinjewadi":     {"coords":[18.5912,73.7389], "weight":0.75},
            "Hadapsar":      {"coords":[18.5018,73.9252], "weight":0.80},
            "Kothrud":       {"coords":[18.5074,73.8077], "weight":0.55},
            "Katraj":        {"coords":[18.4529,73.8567], "weight":0.25},
            "Baner":         {"coords":[18.5590,73.7868], "weight":0.50},
        }
    },
}

AQI_LABELS = {1:"Good",2:"Fair",3:"Moderate",4:"Poor",5:"Very Poor"}
AQI_COLORS = {1:"#27AE60",2:"#A9D18E",3:"#F39C12",4:"#E67E22",5:"#E74C3C"}
AQI_EMOJI  = {1:"🟢",2:"🟡",3:"🟠",4:"🔴",5:"🔴"}

# ── API Functions ─────────────────────────────────────────────
def get_weather(lat, lon):
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        r   = requests.get(url, params={"lat":lat,"lon":lon,"appid":OWM_KEY,"units":"metric"}, timeout=6)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None

def get_aqi(lat, lon):
    try:
        url = "https://api.openweathermap.org/data/2.5/air_pollution"
        r   = requests.get(url, params={"lat":lat,"lon":lon,"appid":OWM_KEY}, timeout=6)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None

def get_aqi_forecast(lat, lon):
    try:
        url = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"
        r   = requests.get(url, params={"lat":lat,"lon":lon,"appid":OWM_KEY}, timeout=8)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None

def calc_health_score(aqi_idx, pm25, pm10, uv=5, temp=30):
    """Indicative project-level Urban Health Score (0-100)."""
    aqi_idx = max(1, min(5, float(aqi_idx)))
    pm25, pm10, uv, temp = max(0.0, float(pm25)), max(0.0, float(pm10)), max(0.0, float(uv)), float(temp)

    aqi_score = max(0, 100 - (aqi_idx - 1) * 22)

    if pm25 <= 12: pm25_score = 100
    elif pm25 <= 35: pm25_score = 80
    elif pm25 <= 55: pm25_score = 55
    elif pm25 <= 150: pm25_score = 30
    else: pm25_score = 10

    if pm10 <= 50: pm10_score = 100
    elif pm10 <= 100: pm10_score = 70
    elif pm10 <= 250: pm10_score = 35
    else: pm10_score = 10

    uv_score = max(0, 100 - uv * 9)
    temp_score = 100 if 20 <= temp <= 28 else 75 if 15 <= temp <= 32 else 50

    score = (aqi_score * 0.40 + pm25_score * 0.25 +
             pm10_score * 0.10 + uv_score * 0.10 + temp_score * 0.15)
    return round(max(0, min(100, score)))

def get_fallback_data(city):
    """Fallback data if API unavailable"""
    defaults = {
        "Bangalore": {"aqi":2,"pm25":22,"pm10":35,"no2":18,"o3":60,"co":400,"temp":28,"humidity":65,"uv":6},
        "Mumbai":    {"aqi":3,"pm25":45,"pm10":72,"no2":35,"o3":55,"co":600,"temp":32,"humidity":75,"uv":7},
        "Delhi":     {"aqi":5,"pm25":180,"pm10":250,"no2":80,"o3":45,"co":1200,"temp":35,"humidity":45,"uv":8},
        "Hyderabad": {"aqi":2,"pm25":28,"pm10":42,"no2":22,"o3":65,"co":450,"temp":31,"humidity":55,"uv":7},
        "Chennai":   {"aqi":2,"pm25":25,"pm10":38,"no2":20,"o3":58,"co":420,"temp":34,"humidity":80,"uv":8},
        "Pune":      {"aqi":2,"pm25":20,"pm10":32,"no2":15,"o3":62,"co":380,"temp":29,"humidity":60,"uv":6},
    }
    return defaults.get(city, defaults["Bangalore"])

# ── Health Zone Map ───────────────────────────────────────────
def build_health_map(city, base_aqi, base_pm25):
    city_info = CITIES[city]
    m = folium.Map(location=city_info["center"],
                   zoom_start=12, tiles="OpenStreetMap")

    from folium.plugins import HeatMap
    heat_data = []

    for zone_name, zone_info in city_info["zones"].items():
        coords = zone_info["coords"]
        weight = zone_info["weight"]

        # Scale AQI per zone by weight
        zone_aqi   = min(5, max(1, round(base_aqi * (0.5 + weight * 0.8))))
        zone_pm25  = round(base_pm25 * (0.5 + weight * 0.8))
        zone_score = calc_health_score(zone_aqi, zone_pm25, zone_pm25 * 1.5, uv=5, temp=30)

        color = AQI_COLORS.get(zone_aqi, "#888")
        emoji = AQI_EMOJI.get(zone_aqi, "⚫")

        popup_html = f"""
        <div style='font-family:Arial;width:170px;padding:4px'>
            <b style='font-size:13px'>{zone_name}</b>
            <hr style='margin:4px 0'>
            Health Score: <b style='color:{color}'>{zone_score}/100</b><br>
            AQI: <b style='color:{color}'>{AQI_LABELS.get(zone_aqi,"--")}</b><br>
            PM2.5: <b>{zone_pm25} µg/m³</b><br>
            Status: {emoji} {AQI_LABELS.get(zone_aqi,"--")}
        </div>"""

        folium.CircleMarker(
            coords, radius=18,
            color=color, fill=True,
            fill_opacity=0.5, weight=2,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{zone_name} — Score: {zone_score}/100"
        ).add_to(m)

        folium.Marker(
            coords,
            icon=folium.DivIcon(
                html=f"""<div style='font-size:9px;font-weight:bold;
                    color:{color};white-space:nowrap;
                    text-shadow:1px 1px 2px white'>{zone_name}</div>""",
                icon_size=(100,14),
                icon_anchor=(50,0)
            )
        ).add_to(m)

        heat_data.append([coords[0], coords[1], weight])

    HeatMap(heat_data, radius=28, blur=20,
            min_opacity=0.15, max_zoom=13).add_to(m)

    # Legend
    legend = f"""
    <div style='position:fixed;bottom:20px;left:20px;z-index:1000;
                background:white;padding:10px;border-radius:8px;
                border:2px solid #1F4E79;font-family:Arial;font-size:11px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.2)'>
        <b style='color:#1F4E79'>Health Index</b><br>
        <span style='color:#27AE60'>●</span> Higher (70-100)<br>
        <span style='color:#F39C12'>●</span> Moderate (50-69)<br>
        <span style='color:#E74C3C'>●</span> Lower (0-49)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m

# ── Charts ────────────────────────────────────────────────────
def aqi_forecast_chart(forecast_data):
    if not forecast_data or "list" not in forecast_data:
        return None
    items  = forecast_data["list"][:24]
    times  = [datetime.datetime.fromtimestamp(item["dt"]).strftime("%H:%M")
               for item in items]
    pm25   = [item.get("components", {}).get("pm2_5", 0) for item in items]
    aqi    = [item.get("main", {}).get("aqi", 1) for item in items]
    colors = [AQI_COLORS.get(a, "#888") for a in aqi]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=times, y=pm25,
        marker_color=colors,
        name="PM2.5",
        hovertemplate="%{x} — PM2.5: %{y} µg/m³<extra></extra>",
    ))
    fig.update_layout(
        height=220, margin=dict(t=20,b=30,l=10,r=10),
        xaxis=dict(title="Time", tickangle=-45, tickfont_size=9),
        yaxis=dict(title="PM2.5 (µg/m³)"),
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

def pollution_bar(components):
    pollutants = {
        "PM2.5": components.get("pm2_5",  0),
        "PM10":  components.get("pm10",   0),
        "NO2":   components.get("no2",    0),
        "O3":    components.get("o3",     0),
        "SO2":   components.get("so2",    0),
        "CO":    components.get("co", 0),
    }
    safe_limits = {"PM2.5":25,"PM10":50,"NO2":40,"O3":100,"SO2":20,"CO":100}
    colors = [
        "#E74C3C" if v > safe_limits.get(k,100) else "#27AE60"
        for k,v in pollutants.items()
    ]
    fig = go.Figure(go.Bar(
        x=list(pollutants.keys()),
        y=list(pollutants.values()),
        marker_color=colors,
        text=[f"{v:.1f}" for v in pollutants.values()],
        textposition="outside",
    ))
    fig.update_layout(
        height=200, margin=dict(t=20,b=20,l=10,r=10),
        xaxis=dict(title="Pollutant"),
        yaxis=dict(title="µg/m³"),
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

def score_gauge(score):
    color = "#27AE60" if score>=70 else "#F39C12" if score>=50 else "#E74C3C"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix":"/100","font":{"size":26,"color":color}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1},
            "bar": {"color":color},
            "steps":[
                {"range":[0,49],  "color":"#FDEDEC"},
                {"range":[49,70], "color":"#FEF9E7"},
                {"range":[70,100],"color":"#EAFAF1"},
            ],
        },
        title={"text":"Urban Health Score","font":{"size":13}},
    ))
    fig.update_layout(height=200,margin=dict(t=40,b=10,l=20,r=20),
                      paper_bgcolor="white")
    return fig

# ── HEADER ───────────────────────────────────────────────────
st.title("🌫️ Urban Health & Pollution Index")
st.markdown("*Air-quality indicators, zone map and an indicative project health score for supported cities*")
st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.title("UrbanAI Navigator")
    st.markdown("*Smart City Life Optimizer*")
    st.markdown("---")
    st.markdown("### Health Index Settings")

    city      = st.selectbox("Select City", list(CITIES.keys()))
    center    = CITIES[city]["center"]
    st.markdown("---")
    fetch_btn = st.button("Fetch Health Data",
                           type="primary", use_container_width=True)
    st.markdown("---")

    if OWM_KEY == "YOUR_KEY_HERE":
        st.warning("OpenWeatherMap API key not set. Using project sample data.")
    else:
        st.success("OpenWeatherMap API Connected")

    st.markdown("---")
    st.markdown("**Project Health Score**")
    st.caption("Indicative composite: air quality 40% + PM2.5 25% + PM10 10% + UV 10% + temperature 15%.")

# ── FETCH DATA ────────────────────────────────────────────────
last_city = st.session_state.get("health_data", {}).get("city")
if fetch_btn or "health_data" not in st.session_state or last_city != city:
    with st.spinner(f"Fetching health data for {city}..."):
        weather_data  = get_weather(center[0], center[1])
        aqi_data      = get_aqi(center[0], center[1])
        forecast_data = get_aqi_forecast(center[0], center[1])

    if aqi_data and aqi_data.get("list") and aqi_data["list"][0].get("components"):
        comp  = aqi_data["list"][0]["components"]
        aqi_i = aqi_data["list"][0].get("main", {}).get("aqi", 3)
        pm25  = comp.get("pm2_5", 0)
        pm10  = comp.get("pm10",  0)
        no2   = comp.get("no2",   0)
        o3    = comp.get("o3",    0)
        so2   = comp.get("so2",   0)
        co    = comp.get("co",    0)
        temp  = weather_data["main"]["temp"]      if weather_data else 30
        hum   = weather_data["main"]["humidity"]  if weather_data else 60
        uv    = 6
        source = "OpenWeatherMap API"
    else:
        fb    = get_fallback_data(city)
        aqi_i = fb["aqi"];  pm25 = fb["pm25"]; pm10  = fb["pm10"]
        no2   = fb["no2"];  o3   = fb["o3"];   so2   = 10;  co = fb["co"]
        temp  = fb["temp"]; hum  = fb["humidity"]; uv = fb["uv"]
        forecast_data = None
        source = "Project sample data (API unavailable)"

    health_score = calc_health_score(aqi_i, pm25, pm10, uv, temp)

    st.session_state["health_data"] = {
        "city":city, "aqi_i":aqi_i, "pm25":pm25, "pm10":pm10,
        "no2":no2, "o3":o3, "so2":so2, "co":co,
        "temp":temp, "hum":hum, "uv":uv,
        "score":health_score, "source":source,
        "forecast":forecast_data,
        "components":{"pm2_5":pm25,"pm10":pm10,"no2":no2,"o3":o3,"so2":so2,"co":co},
    }

# ── Pull from session ─────────────────────────────────────────
d = st.session_state.get("health_data", {})
if not d or d.get("city") != city:
    st.info("Use **Fetch Health Data** in the sidebar to refresh the data.")
    st.stop()

aqi_i        = d["aqi_i"]
pm25         = d["pm25"]
pm10         = d["pm10"]
health_score = d["score"]
temp         = d["temp"]
hum          = d["hum"]
uv           = d["uv"]
source       = d["source"]
forecast_data= d["forecast"]
components   = d["components"]

aqi_label    = AQI_LABELS.get(aqi_i, "Unknown")
aqi_color    = AQI_COLORS.get(aqi_i, "#888")
aqi_emoji    = AQI_EMOJI.get(aqi_i,  "⚫")

# ── TOP METRIC CARDS ──────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Health Score", f"{health_score}/100",
          delta="Good" if health_score>=70 else "Moderate" if health_score>=50 else "Poor")
c2.metric("Air Quality",  f"{aqi_emoji} {aqi_label}",
          delta=f"Index: {aqi_i}/5")
c3.metric("PM2.5",        f"{pm25:.1f} µg/m³",
          delta="Safe" if pm25<25 else "High")
c4.metric("Temperature",  f"{temp:.0f}°C")
c5.metric("Humidity",     f"{hum}%")

st.caption(f"Source: {source} | City: {city} | "
           f"Updated: {datetime.datetime.now().strftime('%H:%M')}")
st.markdown("---")

# ── MAIN CONTENT ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🗺️ Zone Health Map",
    "📈 24-Hour AQI Forecast",
    "🧪 Pollution Breakdown",
])

# ── TAB 1: ZONE MAP ──────────────────────────────────────────
with tab1:
    left, right = st.columns([2, 1])

    with left:
        st.subheader(f"Zone-wise Health Map — {city}")
        with st.spinner("Building health map..."):
            health_map = build_health_map(city, aqi_i, pm25)
        st_folium(health_map, width=700, height=460, returned_objects=[])

    with right:
        # Score gauge
        st.plotly_chart(score_gauge(health_score), use_container_width=True)

        # AQI status card
        css = ("alert-success" if aqi_i <= 2 else
               "alert-warning" if aqi_i == 3 else "alert-danger")
        st.markdown(f"""
        <div class="{css}">
            <h3 style="margin:0;color:{aqi_color}">
                {aqi_emoji} AQI: {aqi_label}
            </h3>
            <p style="margin:4px 0;font-size:12px;color:#555">
                {city} — Index {aqi_i}/5
            </p>
        </div>""", unsafe_allow_html=True)

        # General guidance based on the displayed 1–5 air-quality index.
        st.markdown("**Air-Quality Guidance**")
        if aqi_i >= 4:
            st.error("Higher pollution level — consider reducing prolonged outdoor exposure.")
            st.warning("People sensitive to air pollution may want to take extra precautions.")
        elif aqi_i == 3:
            st.warning("Moderate pollution level — consider limiting prolonged outdoor exposure.")
            st.info("Check local air-quality guidance before strenuous outdoor activity.")
        else:
            st.success("Lower pollution level according to the displayed index.")

        # Zone summary uses the same static project weights as the map.
        zones = CITIES[city]["zones"]
        ranked_zones = sorted(zones.items(), key=lambda item: item[1]["weight"])
        lower_load = [z for z, _ in ranked_zones[:2]]
        higher_load = [z for z, _ in ranked_zones[-2:]][::-1]
        st.info(f"Lower project pollution weight: {', '.join(lower_load)}")
        st.warning(f"Higher project pollution weight: {', '.join(higher_load)}")

# ── TAB 2: 24HR FORECAST ─────────────────────────────────────
with tab2:
    st.subheader(f"24-Hour Air Quality & PM2.5 Forecast — {city}")

    if forecast_data and "list" in forecast_data:
        st.plotly_chart(aqi_forecast_chart(forecast_data),
                        use_container_width=True)

        # Forecast table
        items = forecast_data["list"][:24]
        table = []
        for item in items:
            t     = datetime.datetime.fromtimestamp(item["dt"])
            ai    = item["main"]["aqi"]
            pm    = item["components"].get("pm2_5", 0)
            table.append({
                "Time":    t.strftime("%d %b %H:%M"),
                    "Air quality": f"{AQI_EMOJI.get(ai,'⚫')} {AQI_LABELS.get(ai,'--')}",
                "PM2.5":   f"{pm:.1f} µg/m³",
                "PM10":    f"{item['components'].get('pm10',0):.1f} µg/m³",
                "Guidance": ("Lower pollution level" if ai<=2
                             else "Consider limiting prolonged outdoor exposure" if ai==3
                             else "Consider reducing prolonged outdoor exposure"),
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True,
                     hide_index=True, height=280)
    else:
        # Deterministic estimate when the forecast endpoint is unavailable.
        st.info("Forecast API unavailable — showing a simple estimated pattern from current values.")
        hours = list(range(24))
        pm_vals = [round(max(0, pm25 * (0.82 + 0.18*np.sin(h/4))), 1) for h in hours]
        ai_vals = [min(5, max(1, round(aqi_i * (0.85 + 0.15*np.sin(h/3))))) for h in hours]

        fig = go.Figure(go.Bar(
            x=[f"{h:02d}:00" for h in hours],
            y=pm_vals,
            marker_color=[AQI_COLORS.get(a,"#888") for a in ai_vals],
            text=[f"{v}" for v in pm_vals],
            textposition="outside",
        ))
        fig.update_layout(
            height=240, margin=dict(t=20,b=30,l=10,r=10),
            xaxis=dict(tickangle=-45, tickfont_size=9),
            yaxis=dict(title="PM2.5 (µg/m³)"),
            showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: POLLUTION BREAKDOWN ───────────────────────────────
with tab3:
    st.subheader(f"Pollution Breakdown — {city}")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.plotly_chart(pollution_bar(components),
                        use_container_width=True)

        # Pollutant table
        poll_data = [
            {"Pollutant":"PM2.5","Value":f"{pm25:.1f} µg/m³","Reference":"25 µg/m³",
             "Status":"Within reference" if pm25<25 else "Above reference"},
            {"Pollutant":"PM10", "Value":f"{pm10:.1f} µg/m³","Reference":"50 µg/m³",
             "Status":"Within reference" if pm10<50 else "Above reference"},
            {"Pollutant":"NO2",  "Value":f"{d['no2']:.1f} µg/m³","Reference":"40 µg/m³",
             "Status":"Within reference" if d['no2']<40 else "Above reference"},
            {"Pollutant":"O3",   "Value":f"{d['o3']:.1f} µg/m³","Reference":"100 µg/m³",
             "Status":"Within reference" if d['o3']<100 else "Above reference"},
            {"Pollutant":"SO2",  "Value":f"{d['so2']:.1f} µg/m³","Reference":"20 µg/m³",
             "Status":"Within reference" if d['so2']<20 else "Above reference"},
            {"Pollutant":"CO",   "Value":f"{d['co']:.0f} µg/m³","Reference":"1000 µg/m³",
             "Status":"Within reference" if d['co']<1000 else "Above reference"},
        ]
        st.dataframe(pd.DataFrame(poll_data), use_container_width=True,
                     hide_index=True)

    with col2:
        st.markdown("**Weather Conditions**")
        w1,w2 = st.columns(2)
        w1.metric("Temperature",  f"{temp:.0f}°C")
        w2.metric("Humidity",     f"{hum}%")
        w3,w4 = st.columns(2)
        w3.metric("UV (project default)", uv)
        w4.metric("Health Score", f"{health_score}/100")

        st.markdown("---")
        st.markdown("**General Guidance**")
        groups = [
            ("👶 Children", "Use local air-quality guidance for prolonged outdoor activity."),
            ("🧓 Elderly", "Consider local air-quality guidance before prolonged outdoor activity."),
            ("🏃 Outdoor exercise", "Consider reducing prolonged or strenuous activity at higher pollution levels."),
            ("🌿 Sensitive people", "People sensitive to air pollution may want extra precautions."),
            ("💼 General", "Use the displayed air-quality level as a planning indicator."),
        ]
        for group, advice in groups:
            color = "#E74C3C" if aqi_i >= 4 else "#F39C12" if aqi_i == 3 else "#27AE60"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:12px;margin-bottom:6px'>"
                f"<span>{group}</span>"
                f"<span style='color:{color};font-weight:bold'>{advice}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#888'>UrbanAI Navigator | Urban Health & Pollution Index | "
    "Data: OpenWeatherMap API | VTU Final Year Project | GEC Karwar</small></center>",
    unsafe_allow_html=True
)

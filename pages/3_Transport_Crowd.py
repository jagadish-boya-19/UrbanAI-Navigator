import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
import datetime
from pathlib import Path
from streamlit_folium import st_folium
import plotly.graph_objects as go

st.set_page_config(
    page_title="Transport Crowd - UrbanAI Navigator",
    layout="wide",
    page_icon="🚌"
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
    .crowd-High {
        background:#FDEDEC; border-left:5px solid #E74C3C;
        padding:16px 20px; border-radius:8px; margin:8px 0;
    }
    .crowd-Medium {
        background:#FEF9E7; border-left:5px solid #F39C12;
        padding:16px 20px; border-radius:8px; margin:8px 0;
    }
    .crowd-Low {
        background:#EAFAF1; border-left:5px solid #27AE60;
        padding:16px 20px; border-radius:8px; margin:8px 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ── Station coordinates ───────────────────────────────────────
STATION_COORDS = {
    "Rajiv Chowk":      [28.6327, 77.2190],
    "Hauz Khas":        [28.5450, 77.2066],
    "Kashmere Gate":    [28.6674, 77.2280],
    "Noida Sector 62":  [28.6270, 77.3675],
    "Anand Vihar ISBT": [28.6469, 77.3156],
    "Majestic (KSR)":   [12.9770, 77.5713],
    "Indiranagar":      [12.9784, 77.6408],
    "Silk Board":       [12.9172, 77.6231],
    "Marathahalli":     [12.9591, 77.6971],
    "Hebbal":           [13.0350, 77.5970],
    "Ghatkopar":        [19.0858, 72.9081],
    "Andheri":          [19.1136, 72.8697],
    "Dadar":            [19.0178, 72.8478],
    "Bandra West":      [19.0544, 72.8322],
    "Borivali Station": [19.2307, 72.8567],
    "Hitech City":      [17.4435, 78.3772],
    "Ameerpet":         [17.4375, 78.4482],
    "Secunderabad":     [17.4399, 78.4983],
    "Koyambedu":        [13.0694, 80.1948],
    "T Nagar":          [13.0418, 80.2341],
}

CITY_CENTERS = {
    "Bangalore": [12.9716, 77.5946],
    "Delhi":     [28.6139, 77.2090],
    "Mumbai":    [19.0760, 72.8777],
    "Hyderabad": [17.3850, 78.4867],
    "Chennai":   [13.0827, 80.2707],
}

COLOR_MAP  = {"High":"#E74C3C","Medium":"#F39C12","Low":"#27AE60"}
FOLIUM_CLR = {"High":"red","Medium":"orange","Low":"green"}
EMOJI_MAP  = {"High":"🔴","Medium":"🟠","Low":"🟢"}

# ── Load model and data ───────────────────────────────────────
@st.cache_resource
def load_model():
    base = Path(__file__).resolve().parent.parent
    try:
        with open(base / "models" / "transport_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(base / "models" / "transport_features.pkl", "rb") as f:
            features = pickle.load(f)
        return model, features
    except (FileNotFoundError, OSError, pickle.PickleError, EOFError, AttributeError):
        return None, None

@st.cache_data
def load_station_map():
    base = Path(__file__).resolve().parent.parent
    try:
        df = pd.read_csv(base / "data" / "station_mapping.csv")
        required = {"Station_Code", "Station_Name", "City", "Transit_Type"}
        if not required.issubset(df.columns):
            return None
        df["Station_Code"] = pd.to_numeric(df["Station_Code"], errors="raise").astype(int)
        return df
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError,
            ValueError, TypeError):
        return None

model, features   = load_model()
station_map       = load_station_map()

# ── Prediction function ───────────────────────────────────────
def predict_crowd(station_code, hour, day_idx, month,
                  is_weekend, is_holiday, is_raining):
    if model is None:
        return 50, "Medium"
    inp = pd.DataFrame([{
        "Station_Code": station_code,
        "Month":        month,
        "Hour_Of_Day":  hour,
        "Day_Of_Week":  day_idx,
        "Is_Weekend":   is_weekend,
        "Is_Holiday":   is_holiday,
        "Is_Raining":   is_raining,
    }])[features]
    pct   = int(np.clip(model.predict(inp)[0], 0, 100))
    level = "High" if pct >= 75 else "Medium" if pct >= 40 else "Low"
    return pct, level

def hourly_forecast(station_code, day_idx, month, is_holiday, is_raining):
    pcts, levels = [], []
    for h in range(24):
        is_wknd = 1 if day_idx >= 5 else 0
        pct, lvl = predict_crowd(station_code, h, day_idx, month,
                                  is_wknd, is_holiday, is_raining)
        pcts.append(pct)
        levels.append(lvl)
    return pcts, levels

def weekly_forecast(station_code, start_date, hour, is_raining):
    result = []
    for offset in range(7):
        d = start_date + datetime.timedelta(days=offset)
        day_idx = d.weekday()
        month = d.month
        is_wknd = 1 if day_idx >= 5 else 0
        pct, lvl = predict_crowd(
            station_code, hour, day_idx, month, is_wknd, 0, is_raining
        )
        result.append({"date": d, "day": d.strftime("%a"),
                       "pct": pct, "level": lvl})
    return result

# ── Charts ────────────────────────────────────────────────────
def crowd_gauge(pct, level):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix":"%","font":{"size":28}},
        gauge={
            "axis": {"range":[0,100],"tickwidth":1},
            "bar":  {"color": COLOR_MAP[level]},
            "steps":[
                {"range":[0,40],  "color":"#EAFAF1"},
                {"range":[40,75], "color":"#FEF9E7"},
                {"range":[75,100],"color":"#FDEDEC"},
            ],
            "threshold":{
                "line":{"color":COLOR_MAP[level],"width":4},
                "thickness":0.75,"value":pct,
            },
        },
        title={"text":"Crowd Density","font":{"size":14}},
    ))
    fig.update_layout(height=200, margin=dict(t=40,b=10,l=20,r=20),
                      paper_bgcolor="white")
    return fig

def hourly_chart(pcts, levels, sel_hour):
    colors = [COLOR_MAP[l] for l in levels]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(24)), y=pcts,
        marker_color=colors,
        text=[f"{p}%" for p in pcts],
        textposition="outside",
        hovertemplate="Hour %{x}:00 — %{y}% crowd<extra></extra>",
    ))
    fig.add_vline(x=sel_hour, line_dash="dash", line_color="#1F4E79",
                  line_width=2,
                  annotation_text=f"Selected ({sel_hour}:00)",
                  annotation_font_size=10)
    fig.update_layout(
        height=230, margin=dict(t=30,b=30,l=10,r=10),
        xaxis=dict(title="Hour of Day", dtick=2, tickfont_size=10),
        yaxis=dict(title="Crowd %", range=[0,115]),
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

def weekly_chart(weekly):
    fig = go.Figure(go.Bar(
        x=[w["day"] for w in weekly],
        y=[w["pct"] for w in weekly],
        marker_color=[COLOR_MAP[w["level"]] for w in weekly],
        text=[f"{w['pct']}%" for w in weekly],
        textposition="outside",
        hovertemplate="%{x} — %{y}% crowd<extra></extra>",
    ))
    fig.update_layout(
        height=210, margin=dict(t=20,b=20,l=10,r=10),
        xaxis=dict(title="Day of Week", tickfont_size=10),
        yaxis=dict(title="Crowd %", range=[0,115]),
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

# ── Station map ───────────────────────────────────────────────
def build_station_map(city, station_name, transit_type, level):
    center = CITY_CENTERS.get(city, [12.9716, 77.5946])
    coords = STATION_COORDS.get(station_name, center)
    m = folium.Map(location=coords, zoom_start=14, tiles="OpenStreetMap")
    folium.CircleMarker(
        coords, radius=60, color=COLOR_MAP[level],
        fill=True, fill_opacity=0.1, weight=1
    ).add_to(m)
    folium.CircleMarker(
        coords, radius=30, color=COLOR_MAP[level],
        fill=True, fill_opacity=0.2, weight=2
    ).add_to(m)
    icon_name = "train" if "metro" in str(transit_type).lower() else "bus"
    folium.Marker(
        coords,
        popup=folium.Popup(
            f"<b>{station_name}</b><br>"
            f"Crowd: <b style='color:{COLOR_MAP[level]}'>{level}</b><br>"
            f"City: {city}<br>Mode: {transit_type}",
            max_width=200
        ),
        tooltip=f"{station_name} — {level} crowd",
        icon=folium.Icon(color=FOLIUM_CLR[level], icon=icon_name, prefix="glyphicon")
    ).add_to(m)
    return m

# ── HEADER ───────────────────────────────────────────────────
st.title("🚌 Transport Crowd Forecaster")
st.markdown("*AI-based transport crowd prediction with 24-hour and 7-day forecasts*")
st.markdown("---")

# ── Model status ─────────────────────────────────────────────
if model is None or station_map is None:
    st.error("""
    Model or station data not found! Run these first:
    ```
    python data_generator.py
    python train_transport.py
    ```
    """)
    st.stop()
else:
    st.success(f"Model loaded — Random Forest | R²: 99.33% | "
               f"{station_map['Station_Name'].nunique()} stations | "
               f"{station_map['City'].nunique()} cities")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.title("UrbanAI Navigator")
    st.markdown("*Smart City Life Optimizer*")
    st.markdown("---")
    st.markdown("### Transport Settings")

    cities         = sorted(station_map["City"].unique())
    city           = st.selectbox("Select City", cities)
    filtered       = station_map[station_map["City"] == city]
    station_name   = st.selectbox("Station / Hub",
                                   sorted(filtered["Station_Name"].unique()))
    station_row    = filtered[filtered["Station_Name"] == station_name].iloc[0]
    station_code   = int(station_row["Station_Code"])
    transit_type   = station_row["Transit_Type"]

    st.markdown(f"**Mode:** {transit_type}")
    st.markdown("---")

    today = datetime.date.today()
    selected_date = st.date_input("Select Date", value=today)
    selected_time = st.time_input(
        "Select Time",
        value=datetime.time(datetime.datetime.now().hour, 0)
    )
    hour = selected_time.hour
    day_idx = selected_date.weekday()
    month = selected_date.month
    is_weekend = 1 if day_idx >= 5 else 0

    st.markdown("---")
    is_raining = 1 if st.toggle("Is it raining?", value=False) else 0
    is_holiday = 1 if st.toggle("Public holiday?", value=False) else 0
    is_weekend = 1 if day_idx >= 5 else 0
    st.markdown("---")
    predict_btn = st.button("Forecast Crowd Level", type="primary",
                             use_container_width=True)
    st.markdown("---")
    st.info(f"Model: Random Forest\nR²: 99.33%\nDataset: 175,200 rows\n5 Indian cities")

# ── TOP METRICS ───────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("City",    city)
c2.metric("Station", station_name)
c3.metric("Mode",    transit_type)
c4.metric("Time",    f"{hour}:00 | {'Weekend' if is_weekend else 'Weekday'}")
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📍 Current Forecast",
    "📈 24-Hour Forecast",
    "📅 Weekly Forecast",
])

# ── TAB 1: CURRENT ────────────────────────────────────────────
with tab1:
    if not predict_btn:
        st.info("Select a city and station in the sidebar, then click **Forecast Crowd Level**.")
        # Default city map with all stations
        center = CITY_CENTERS.get(city, [12.9716, 77.5946])
        m_def  = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")
        for _, row in filtered.iterrows():
            sname  = row["Station_Name"]
            coords = STATION_COORDS.get(sname, center)
            folium.Marker(
                coords, tooltip=sname, popup=sname,
                icon=folium.Icon(color="blue", icon="map-marker", prefix="glyphicon")
            ).add_to(m_def)
        st_folium(m_def, width=1100, height=430, returned_objects=[])

    else:
        pct, level = predict_crowd(station_code, hour, day_idx,
                                    month, is_weekend, is_holiday, is_raining)

        # Update metrics
        c1.metric("City",    city)
        c2.metric("Station", station_name)
        c3.metric("Crowd",   f"{pct}%",
                  delta=level)
        c4.metric("Status",  EMOJI_MAP[level] + " " + level)

        left, middle, right = st.columns([1.2, 1, 1])

        with left:
            st.markdown(f"""
            <div class="crowd-{level}">
                <h2 style="margin:0;color:{COLOR_MAP[level]}">
                    {EMOJI_MAP[level]} {level} Crowd — {pct}%
                </h2>
                <p style="margin:6px 0 0;font-size:13px;color:#555">
                    {station_name}, {city} &nbsp;|&nbsp;
                    {transit_type} &nbsp;|&nbsp;
                    {hour}:00 &nbsp;|&nbsp;
                    {'Raining' if is_raining else 'Clear'}
                </p>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if level == "High":
                st.error("**High predicted crowd level**")
                st.warning("Allow extra travel time and consider less busy hours.")
            elif level == "Medium":
                st.warning("**Medium predicted crowd level**")
                st.info("Moderate crowd is predicted for the selected time.")
            else:
                st.success("**Low predicted crowd level**")
                st.info("A lower crowd level is predicted for the selected time.")

            st.caption(
                "Advice is based on predicted crowd level; actual passenger volume "
                "and service delays may differ."
            )

            st.markdown("---")
            st.markdown("**Lower-crowd hours today**")
            pcts_today, _ = hourly_forecast(
                station_code, day_idx, month, is_holiday, is_raining
            )
            for h in sorted(range(24), key=lambda x: pcts_today[x])[:5]:
                a1, b1 = st.columns([2, 1])
                a1.markdown(f"<small>{h:02d}:00</small>", unsafe_allow_html=True)
                b1.markdown(
                    f"<small><b>{pcts_today[h]}%</b></small>",
                    unsafe_allow_html=True
                )

        with middle:
            st.plotly_chart(crowd_gauge(pct, level), use_container_width=True)
            st.markdown("**Quick Info**")
            r1,r2 = st.columns(2)
            r1.metric("Crowd %",  f"{pct}%")
            r2.metric("Level",    level)
            r3,r4 = st.columns(2)
            r3.metric("Weekend",  "Yes" if is_weekend else "No")
            r4.metric("Holiday",  "Yes" if is_holiday else "No")
            r5,r6 = st.columns(2)
            r5.metric("Raining",  "Yes" if is_raining else "No")
            r6.metric("Mode",     transit_type)

        with right:
            st.markdown(f"**{station_name}**")
            st_folium(
                build_station_map(city, station_name, transit_type, level),
                width=310, height=370, returned_objects=[]
            )

# ── TAB 2: 24-HOUR FORECAST ───────────────────────────────────
with tab2:
    st.subheader(f"24-Hour Crowd Forecast — {station_name}, {city}")
    st.caption(f"{'Weekend' if is_weekend else 'Weekday'} | "
               f"{'Raining' if is_raining else 'Clear'} | "
               f"{'Public Holiday' if is_holiday else 'Regular day'}")

    with st.spinner("Generating 24-hour forecast..."):
        pcts_24, levels_24 = hourly_forecast(
            station_code, day_idx, month, is_holiday, is_raining
        )

    high_h   = [i for i,l in enumerate(levels_24) if l=="High"]
    medium_h = [i for i,l in enumerate(levels_24) if l=="Medium"]
    low_h    = [i for i,l in enumerate(levels_24) if l=="Low"]
    best_pct = min(pcts_24)
    best_hr  = pcts_24.index(best_pct)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Peak Hours",    len(high_h),
              delta=f"Starts {high_h[0]}:00" if high_h else "None today")
    c2.metric("Moderate Hours", len(medium_h))
    c3.metric("Low Crowd Hours",len(low_h))
    c4.metric("Best Time",     f"{best_hr}:00",
              delta=f"{best_pct}% crowd")

    st.markdown("---")
    st.plotly_chart(hourly_chart(pcts_24, levels_24, hour),
                    use_container_width=True)

    st.markdown("**Hour-by-Hour Breakdown**")
    table = []
    for i,(pct,lvl) in enumerate(zip(pcts_24, levels_24)):
        period = ("Early Morning" if i < 6
                  else "Morning Peak" if 6 <= i <= 10
                  else "Midday" if 10 < i <= 15
                  else "Evening Peak" if 15 < i <= 20
                  else "Night")
        table.append({
            "Hour":        f"{i:02d}:00",
            "Period":      period,
            "Crowd %":     f"{pct}%",
            "Level":       f"{EMOJI_MAP[lvl]} {lvl}",
            "Advice":      ("Lower predicted crowd" if lvl=="Low"
                            else "Moderate predicted crowd" if lvl=="Medium"
                            else "Higher predicted crowd"),
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True,
                 hide_index=True, height=300)

# ── TAB 3: WEEKLY FORECAST ────────────────────────────────────
with tab3:
    st.subheader(f"7-Day Weekly Forecast — {station_name}, {city}")
    st.caption(f"At {hour}:00 | {'Raining' if is_raining else 'Clear'}")

    with st.spinner("Generating weekly forecast..."):
        weekly = weekly_forecast(station_code, selected_date, hour, is_raining)

    st.plotly_chart(weekly_chart(weekly), use_container_width=True)

    cols = st.columns(7)
    for col, w in zip(cols, weekly):
        with col:
            st.markdown(f"""
            <div style="text-align:center;background:white;border-radius:8px;
                        padding:12px 4px;box-shadow:0 1px 4px rgba(0,0,0,0.08)">
                <div style="font-size:11px;color:#888;margin-bottom:2px">
                    {w["day"]}
                </div>
                <div style="font-size:9px;color:#aaa;margin-bottom:4px">
                    {w["date"].strftime("%d %b")}
                </div>
                <div style="font-size:22px">{EMOJI_MAP[w["level"]]}</div>
                <div style="font-size:13px;font-weight:bold;
                            color:{COLOR_MAP[w["level"]]}">
                    {w["pct"]}%
                </div>
                <div style="font-size:9px;color:{COLOR_MAP[w["level"]]};
                            font-weight:bold;margin-top:2px">
                    {w["level"]}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    best_days  = [w["day"] for w in weekly if w["level"]=="Low"]
    worst_days = [w["day"] for w in weekly if w["level"]=="High"]
    col1, col2 = st.columns(2)
    with col1:
        if best_days:
            st.success(f"**Lower predicted crowd:** {', '.join(best_days)}")
        else:
            st.warning("No low-crowd days at this hour this week")
    with col2:
        if worst_days:
            st.error(f"**Higher predicted crowd:** {', '.join(worst_days)}")
        else:
            st.success("No severely crowded days this week!")

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#888'>UrbanAI Navigator | Transport Crowd Forecaster | "
    "Model: Random Forest | R²: 99.33% | Dataset: 175,200 rows | "
    "VTU Final Year Project 2025-26 | GEC Karwar</small></center>",
    unsafe_allow_html=True
)

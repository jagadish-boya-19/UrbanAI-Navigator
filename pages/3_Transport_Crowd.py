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
    .stMetric { background:white; padding:12px; border-radius:10px;
        border-left:4px solid #1F4E79;
        box-shadow:0 2px 8px rgba(0,0,0,0.08); }
    .stButton > button { background-color:#1F4E79; color:white;
        border-radius:8px; border:none; font-weight:bold; width:100%; }
    .stButton > button:hover { background-color:#2E86AB; }
    .crowd-High { background:#FDEDEC; border-left:5px solid #E74C3C;
        padding:16px 20px; border-radius:8px; margin:8px 0; }
    .crowd-Medium { background:#FEF9E7; border-left:5px solid #F39C12;
        padding:16px 20px; border-radius:8px; margin:8px 0; }
    .crowd-Low { background:#EAFAF1; border-left:5px solid #27AE60;
        padding:16px 20px; border-radius:8px; margin:8px 0; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"

STATION_COORDS = {
    "Rajiv Chowk":[28.6327,77.2190],"Hauz Khas":[28.5450,77.2066],
    "Kashmere Gate":[28.6674,77.2280],"Noida Sector 62":[28.6270,77.3675],
    "Anand Vihar ISBT":[28.6469,77.3156],
    "Majestic (KSR)":[12.9770,77.5713],"Indiranagar":[12.9784,77.6408],
    "Silk Board":[12.9172,77.6231],"Marathahalli":[12.9591,77.6971],
    "Hebbal":[13.0350,77.5970],
    "Ghatkopar":[19.0858,72.9081],"Andheri":[19.1136,72.8697],
    "Dadar":[19.0178,72.8478],"Bandra West":[19.0544,72.8322],
    "Borivali Station":[19.2307,72.8567],
    "Hitech City":[17.4435,78.3772],"Ameerpet":[17.4375,78.4482],
    "Secunderabad":[17.4399,78.4983],
    "Koyambedu":[13.0694,80.1948],"T Nagar":[13.0418,80.2341],
}

CITY_CENTERS = {
    "Bangalore":[12.9716,77.5946],"Delhi":[28.6139,77.2090],
    "Mumbai":[19.0760,72.8777],"Hyderabad":[17.3850,78.4867],
    "Chennai":[13.0827,80.2707],
}

COLOR_MAP  = {"High":"#E74C3C","Medium":"#F39C12","Low":"#27AE60"}
FOLIUM_CLR = {"High":"red","Medium":"orange","Low":"green"}
EMOJI_MAP  = {"High":"🔴","Medium":"🟠","Low":"🟢"}
TRANSIT_ICON = {"Metro":"transfer","Bus":"road","Train":"train"}

@st.cache_resource
def load_model():
    mp = MODEL_DIR / "transport_model.pkl"
    fp = MODEL_DIR / "transport_features.pkl"
    missing = [str(p) for p in [mp,fp] if not p.exists()]
    if missing:
        return None, None, f"Missing: {', '.join(missing)}"
    try:
        m = pickle.load(open(mp,"rb"))
        f = pickle.load(open(fp,"rb"))
        return m, f, None
    except pickle.UnpicklingError as e:
        return None, None, f"Corrupted model: {e}"
    except Exception as e:
        return None, None, f"Load error: {e}"

@st.cache_data
def load_station_map():
    csv = DATA_DIR / "station_mapping.csv"
    if not csv.exists():
        return None, f"station_mapping.csv not found at {csv}"
    try:
        df = pd.read_csv(csv)
        df["Station_Code"] = df["Station_Code"].astype(int)
        return df, None
    except (pd.errors.ParserError, KeyError, ValueError) as e:
        return None, f"CSV error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"

model, features, model_err    = load_model()
station_map, station_map_err  = load_station_map()

def predict_crowd(station_code, hour, day_idx, month,
                  is_weekend, is_holiday, is_raining):
    if model is None:
        return 50, "Medium"
    try:
        inp = pd.DataFrame([{
            "Station_Code":station_code,"Month":month,
            "Hour_Of_Day":hour,"Day_Of_Week":day_idx,
            "Is_Weekend":is_weekend,"Is_Holiday":is_holiday,
            "Is_Raining":is_raining,
        }])[features]
        pct   = int(np.clip(model.predict(inp)[0], 0, 100))
        level = "High" if pct>=75 else "Medium" if pct>=40 else "Low"
        return pct, level
    except Exception as e:
        st.warning(f"Prediction error: {e}")
        return 50, "Medium"

def hourly_forecast(station_code, day_idx, month, is_holiday, is_raining):
    pcts, levels = [], []
    for h in range(24):
        is_wknd = 1 if day_idx>=5 else 0
        p, l = predict_crowd(station_code, h, day_idx, month,
                              is_wknd, is_holiday, is_raining)
        pcts.append(p); levels.append(l)
    return pcts, levels

def next7_forecast(station_code, start_date, hour, is_raining):
    results = []
    for i in range(7):
        d = start_date + datetime.timedelta(days=i)
        di = d.weekday()
        is_wknd = 1 if di >= 5 else 0

        # A holiday calendar is not present in the current data files.
        # Future dates therefore use the regular-day setting.
        holiday_for_date = 0
        p, l = predict_crowd(
            station_code, hour, di, d.month,
            is_wknd, holiday_for_date, is_raining
        )
        results.append({"date_label":d.strftime("%a %d"),
                        "day_idx":di,"pct":p,"level":l})
    return results

def crowd_gauge(pct, level):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={"suffix":"%","font":{"size":28}},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":COLOR_MAP[level]},
            "steps":[
                {"range":[0,40],"color":"#EAFAF1"},
                {"range":[40,75],"color":"#FEF9E7"},
                {"range":[75,100],"color":"#FDEDEC"},
            ],
            "threshold":{"line":{"color":COLOR_MAP[level],"width":4},
                         "thickness":0.75,"value":pct},
        },
        title={"text":"Crowd Density","font":{"size":14}},
    ))
    fig.update_layout(height=200,margin=dict(t=40,b=10,l=20,r=20),
                      paper_bgcolor="white")
    return fig

def hourly_chart(pcts, levels, sel_hour, use_now):
    ann = f"Current ({sel_hour}:00)" if use_now else f"Selected ({sel_hour}:00)"
    fig = go.Figure(go.Bar(
        x=list(range(24)), y=pcts,
        marker_color=[COLOR_MAP[l] for l in levels],
        text=[f"{p}%" for p in pcts], textposition="outside",
        hovertemplate="Hour %{x}:00 — %{y}%<extra></extra>",
    ))
    fig.add_vline(x=sel_hour, line_dash="dash",
                  line_color="#1F4E79", line_width=2,
                  annotation_text=ann, annotation_font_size=10)
    fig.update_layout(height=230, margin=dict(t=30,b=30,l=10,r=10),
        xaxis=dict(title="Hour of Day",dtick=2,tickfont_size=10),
        yaxis=dict(title="Crowd %",range=[0,115]),
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    return fig

def weekly_chart(weekly):
    fig = go.Figure(go.Bar(
        x=[w["date_label"] for w in weekly],
        y=[w["pct"] for w in weekly],
        marker_color=[COLOR_MAP[w["level"]] for w in weekly],
        text=[f"{w['pct']}%" for w in weekly], textposition="outside",
        hovertemplate="%{x} — %{y}%<extra></extra>",
    ))
    fig.update_layout(height=210, margin=dict(t=20,b=20,l=10,r=10),
        xaxis=dict(title="Date",tickfont_size=10),
        yaxis=dict(title="Crowd %",range=[0,115]),
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    return fig

def build_station_map(city, station_name, transit_type, level):
    center = CITY_CENTERS.get(city,[12.9716,77.5946])
    coords = STATION_COORDS.get(station_name)
    if coords is None:
        st.caption(f"Exact coordinates for {station_name} not on file — showing city centre.")
        coords = center
    m = folium.Map(location=coords, zoom_start=14, tiles="OpenStreetMap")
    folium.CircleMarker(coords, radius=60, color=COLOR_MAP[level],
                        fill=True, fill_opacity=0.1, weight=1).add_to(m)
    folium.CircleMarker(coords, radius=30, color=COLOR_MAP[level],
                        fill=True, fill_opacity=0.2, weight=2).add_to(m)
    icon_name = TRANSIT_ICON.get(transit_type, "map-marker")
    folium.Marker(coords,
        popup=folium.Popup(
            f"<b>{station_name}</b><br>Type: {transit_type}<br>"
            f"Crowd: <b style='color:{COLOR_MAP[level]}'>{level}</b>",
            max_width=180),
        tooltip=f"{station_name} ({transit_type}) — {level}",
        icon=folium.Icon(color=FOLIUM_CLR[level],
                         icon=icon_name, prefix="glyphicon")
    ).add_to(m)
    return m

# ── HEADER ──────────────────────────────────────────────────
st.title("🚌 Transport Crowd Forecaster")
st.markdown("*Predict expected crowd levels for your selected transport hub and travel time.*")
st.markdown("---")

if model_err:
    st.error(f"Model unavailable: {model_err}. Run data_generator.py then train_transport.py.", icon="🚨")
if station_map_err:
    st.error(f"Station data unavailable: {station_map_err}", icon="🚨")
if model_err or station_map_err:
    st.stop()

st.success(
    f"Model loaded — Random Forest | R\u00b2 Score: 0.9933 | "
    f"{station_map['Station_Name'].nunique()} stations | "
    f"{station_map['City'].nunique()} cities"
)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.title("UrbanAI Navigator")
    st.markdown("*Smart City Life Optimizer*")
    st.markdown("---")
    st.markdown("### Transport Settings")

    cities = sorted(station_map["City"].unique())
    city   = st.selectbox("Select City", cities)

    all_types      = sorted(station_map[station_map["City"]==city]["Transit_Type"].unique())
    transit_filter = st.selectbox("Transport Type", ["All"]+all_types)

    filtered = station_map[station_map["City"]==city]
    if transit_filter != "All":
        filtered = filtered[filtered["Transit_Type"]==transit_filter]

    if filtered.empty:
        st.warning(f"No {transit_filter} stations in {city}.")
        st.stop()

    station_name = st.selectbox("Station / Hub", sorted(filtered["Station_Name"].dropna().unique()))
    station_row  = filtered[filtered["Station_Name"]==station_name].iloc[0]
    station_code = int(station_row["Station_Code"])
    st.caption(f"{len(filtered)} {transit_filter.lower() if transit_filter != 'All' else 'transport'} hubs available in {city}")
    transit_type = station_row["Transit_Type"]
    st.markdown(f"**Mode:** {transit_type}")
    st.markdown("---")

    now = datetime.datetime.now()
    selected_date = st.date_input(
        "Travel Date",
        value=now.date(),
        min_value=now.date()
    )
    hour = st.slider("Travel Time", 0, 23, now.hour, format="%02d:00")
    day_idx = selected_date.weekday()
    month = selected_date.month
    use_now = selected_date == now.date() and hour == now.hour

    st.caption(
        f"{selected_date.strftime('%A, %d %B %Y')} | "
        f"{'Current time' if use_now else f'{hour:02d}:00'}"
    )

    st.markdown("---")
    st.markdown("**Prediction Inputs**")
    is_raining = 1 if st.toggle("Raining?", value=False) else 0
    is_holiday = 1 if st.toggle("Public Holiday?", value=False) else 0
    is_weekend = 1 if day_idx>=5 else 0
    st.markdown("---")
    predict_btn = st.button("Forecast Crowd Level", type="primary", use_container_width=True)
    st.markdown("---")
    st.info("Choose your travel date, time and transport hub, then forecast the expected crowd level.")

# ── PREDICTION ──────────────────────────────────────────────
if predict_btn:
    pct, level = predict_crowd(station_code, hour, day_idx, month,
                                is_weekend, is_holiday, is_raining)
    st.session_state.update({
        "tc_done":True,"tc_pct":pct,"tc_level":level,
        "tc_city":city,"tc_station":station_name,
        "tc_transit":transit_type,"tc_code":station_code,
        "tc_date":selected_date,
        "tc_hour":hour,"tc_day_idx":day_idx,"tc_month":month,
        "tc_weekend":is_weekend,"tc_holiday":is_holiday,
        "tc_raining":is_raining,"tc_use_now":use_now,
    })
    st.rerun()

result_ready = st.session_state.get("tc_done", False)
if result_ready:
    pct        = st.session_state["tc_pct"]
    level      = st.session_state["tc_level"]
    r_city     = st.session_state["tc_city"]
    r_station  = st.session_state["tc_station"]
    r_transit  = st.session_state["tc_transit"]
    r_code     = st.session_state["tc_code"]
    r_date     = st.session_state["tc_date"]
    r_hour     = st.session_state["tc_hour"]
    r_day_idx  = st.session_state["tc_day_idx"]
    r_month    = st.session_state["tc_month"]
    r_weekend  = st.session_state["tc_weekend"]
    r_holiday  = st.session_state["tc_holiday"]
    r_raining  = st.session_state["tc_raining"]
    r_use_now  = st.session_state["tc_use_now"]

# ── TOP METRICS ─────────────────────────────────────────────
m1,m2,m3,m4 = st.columns(4)
m1.metric("City", city)
m2.metric("Station", station_name)
m3.metric("Mode", transit_type)
m4.metric("Crowd",
    f"{pct}% — {level}" if result_ready else "--",
    delta=("Peak hour" if (7<=hour<=10 or 17<=hour<=20) else "Off-peak")
    if result_ready else None)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📍 Current Forecast","📈 24-Hour Forecast","📅 7-Day Forecast"])

with tab1:
    if not result_ready:
        st.info("Select city, transport type and station, then click Forecast Crowd Level.")
        center = CITY_CENTERS.get(city,[12.9716,77.5946])
        m_def  = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")
        for _, row in filtered.iterrows():
            sname  = row["Station_Name"]
            coords = STATION_COORDS.get(sname, center)
            folium.Marker(coords, tooltip=f"{sname} ({row['Transit_Type']})",
                popup=sname,
                icon=folium.Icon(color="blue",
                    icon=TRANSIT_ICON.get(row["Transit_Type"],"map-marker"),
                    prefix="glyphicon")
            ).add_to(m_def)
        st_folium(m_def, width=1100, height=430, returned_objects=[])
    else:
        left, middle, right = st.columns([1.2,1,1])
        with left:
            st.markdown(f"""<div class="crowd-{level}">
                <h2 style="margin:0;color:{COLOR_MAP[level]}">{EMOJI_MAP[level]} {level} Crowd — {pct}%</h2>
                <p style="margin:6px 0 0;font-size:13px;color:#555">
                {r_station}, {r_city} | {r_transit} | {r_date.strftime('%d %b %Y')} | {r_hour:02d}:00 |
                {"Raining" if r_raining else "Clear"}</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if level=="High":
                st.error("**High crowd expected**")
                st.info("Try a lower-crowd time if your schedule is flexible.")
                st.warning("Another transport option may be more comfortable.")
            elif level=="Medium":
                st.warning("**Moderately crowded**")
                st.info("Some crowding is expected — allow a little extra time.")
            else:
                st.success("**Low crowd expected**")
                st.info("A relatively comfortable time to travel.")
            st.markdown("<small style='color:#888'>Prediction is based on the selected date, time and available station data; it is not a live crowd feed.</small>",
                        unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**Travel Advice**")
            adv_pcts, adv_levels = hourly_forecast(r_code, r_day_idx, r_month, r_holiday, r_raining)
            low_hrs  = [h for h,l in enumerate(adv_levels) if l=="Low"]
            high_hrs = [h for h,l in enumerate(adv_levels) if l=="High"]
            if low_hrs:
                st.success("Lower crowd: " + ", ".join(f"{h:02d}:00" for h in low_hrs[:4]))
            if high_hrs:
                st.error("Higher crowd: " + ", ".join(f"{h:02d}:00" for h in high_hrs[:4]))
        with middle:
            st.plotly_chart(crowd_gauge(pct, level), use_container_width=True)
            r1,r2 = st.columns(2)
            r1.metric("Weekend","Yes" if r_weekend else "No")
            r2.metric("Holiday","Yes" if r_holiday else "No")
            r3,r4 = st.columns(2)
            r3.metric("Raining","Yes" if r_raining else "No")
            r4.metric("Holiday","Yes" if r_holiday else "No")
        with right:
            st.markdown(f"**{r_station}**")
            st_folium(build_station_map(r_city,r_station,r_transit,level),
                      width=310, height=370, returned_objects=[])

with tab2:
    st.subheader(f"24-Hour Forecast — {station_name}, {city}")
    st.caption(
        f"{selected_date.strftime('%A, %d %b %Y')} | "
        f"{'Raining' if is_raining else 'Clear'} | "
        f"{'Holiday' if is_holiday else 'Regular day'} | {transit_type}"
    )
    pcts_24, levels_24 = hourly_forecast(station_code, day_idx, month, is_holiday, is_raining)
    high_h = [i for i,l in enumerate(levels_24) if l=="High"]
    low_h  = [i for i,l in enumerate(levels_24) if l=="Low"]
    mid_h  = [i for i,l in enumerate(levels_24) if l=="Medium"]
    best_hr= pcts_24.index(min(pcts_24))
    f1,f2,f3,f4 = st.columns(4)
    f1.metric("Highest Crowd", f"{max(pcts_24)}%")
    f2.metric("Lowest Crowd", f"{min(pcts_24)}%")
    f3.metric("Best Time", f"{best_hr:02d}:00")
    f4.metric("Selected Time", f"{hour:02d}:00")
    st.markdown("---")
    st.plotly_chart(hourly_chart(pcts_24, levels_24, hour, use_now), use_container_width=True)
    st.markdown("**Hour-by-Hour Breakdown**")
    table = []
    for i,(p,lvl) in enumerate(zip(pcts_24,levels_24)):
        period = ("Early Morning" if i<6 else "Morning Peak" if 6<=i<=10
                  else "Midday" if 10<i<=15 else "Evening Peak" if 15<i<=20 else "Night")
        table.append({"Hour":f"{i:02d}:00","Period":period,"Crowd %":f"{p}%",
            "Level":f"{EMOJI_MAP[lvl]} {lvl}",
            "Action":("Lower crowd ✅" if lvl=="Low" else "Moderate ⚠️" if lvl=="Medium" else "Higher crowd ❌")})
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True, height=300)

with tab3:
    st.subheader(f"7-Day Forecast — {station_name}, {city}")
    st.caption(f"7 days from {selected_date.strftime('%d %b %Y')} at {hour:02d}:00 | {transit_type}")
    weekly = next7_forecast(station_code, selected_date, hour, is_raining)
    st.plotly_chart(weekly_chart(weekly), use_container_width=True)
    cols = st.columns(7)
    for col,w in zip(cols,weekly):
        with col:
            st.markdown(f"""<div style="text-align:center;background:white;border-radius:8px;
                padding:10px 4px;box-shadow:0 1px 4px rgba(0,0,0,0.08)">
                <div style="font-size:10px;color:#888;margin-bottom:2px">{w["date_label"]}</div>
                <div style="font-size:22px">{EMOJI_MAP[w["level"]]}</div>
                <div style="font-size:12px;font-weight:bold;color:{COLOR_MAP[w["level"]]};margin-top:2px">
                    {w["pct"]}%</div>
                <div style="font-size:9px;color:{COLOR_MAP[w["level"]]};font-weight:bold">
                    {w["level"]}</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    low_days  = [w["date_label"] for w in weekly if w["level"]=="Low"]
    high_days = [w["date_label"] for w in weekly if w["level"]=="High"]
    c1,c2 = st.columns(2)
    with c1:
        if low_days: st.success(f"**Lower predicted crowd:** {', '.join(low_days)}")
        else: st.warning("No low-crowd days predicted this week")
    with c2:
        if high_days: st.error(f"**Higher predicted crowd:** {', '.join(high_days)}")
        else: st.success("No high-crowd days predicted this week")

st.markdown("---")
st.markdown("<center><small style='color:#888'>UrbanAI Navigator | Transport Crowd Forecaster | "
            "VTU Final Year Project 2025-26 | GEC Karwar</small></center>",
            unsafe_allow_html=True)

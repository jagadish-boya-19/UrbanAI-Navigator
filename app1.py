import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv
import os
import hashlib
import sqlite3
import datetime
import json

load_dotenv()
TOMTOM_KEY = os.getenv("TOMTOM_API_KEY", "JmX9xc6dwCIvEAyAt8fUv9Yg5jJwF0q4")

st.set_page_config(
    page_title="UrbanAI Navigator",
    layout="wide",
    page_icon="🏙️"
)

# ================================================================
# PASSWORD HASHING — PBKDF2 (production-safe)
# ================================================================
def hash_pw(password: str, salt: str = "urbanai_salt_2025") -> str:
    """PBKDF2-HMAC-SHA256 — stronger than plain SHA-256"""
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=260000
    )
    return dk.hex()

# ================================================================
# DATABASE FUNCTIONS
# ================================================================
DB_PATH = "data/users.db"

def db_exists() -> bool:
    return os.path.exists(DB_PATH)

def get_user(username: str):
    if not db_exists():
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            "SELECT username, password, name, role FROM users WHERE username = ?",
            (username.strip().lower(),)
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return {"username": row[0], "password": row[1],
                    "name": row[2], "role": row[3]}
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    return None

def username_exists(username: str) -> bool:
    if not db_exists():
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE username = ?",
                    (username.strip().lower(),))
        exists = cur.fetchone() is not None
        conn.close()
        return exists
    except sqlite3.Error:
        return False

def create_user(username: str, password: str, name: str) -> bool:
    """New users always get Viewer role — admin created separately"""
    if not db_exists():
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password, name, role) VALUES (?,?,?,?)",
            (username.strip().lower(), hash_pw(password), name.strip(), "Viewer")
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return False

# ================================================================
# API KEY VALIDATION
# ================================================================
def validate_api_key() -> tuple[bool, str]:
    """Check TomTom key before making any requests"""
    if not TOMTOM_KEY:
        return False, "TOMTOM_API_KEY not found in .env file"
    if TOMTOM_KEY == "YOUR_API_KEY":
        return False, "Placeholder API key detected — set a real key in .env"
    return True, "OK"

# ================================================================
# TOMTOM FUNCTIONS — No fallback, errors shown clearly
# ================================================================
MANUAL_COORDS = {
    # Bangalore
    "mg road":         (12.9758, 77.6085),
    "silk board":      (12.9172, 77.6231),
    "whitefield":      (12.9698, 77.7500),
    "koramangala":     (12.9352, 77.6245),
    "hebbal":          (13.0350, 77.5970),
    "electronic city": (12.8458, 77.6603),
    "jayanagar":       (12.9308, 77.5830),
    "marathahalli":    (12.9591, 77.6971),
    "indiranagar":     (12.9784, 77.6408),
    "rajajinagar":     (12.9918, 77.5538),
    # Mumbai
    "bandra":          (19.0596, 72.8295),
    "andheri":         (19.1136, 72.8697),
    "dadar":           (19.0178, 72.8478),
    "borivali":        (19.2307, 72.8567),
    "thane":           (19.2183, 72.9781),
    # Delhi
    "connaught place": (28.6315, 77.2167),
    "lajpat nagar":    (28.5700, 77.2433),
    "karol bagh":      (28.6519, 77.1907),
    "dwarka":          (28.5921, 77.0460),
    # Hyderabad
    "hitech city":     (17.4435, 78.3772),
    "charminar":       (17.3616, 78.4747),
    "gachibowli":      (17.4401, 78.3489),
    # Chennai
    "t nagar":         (13.0418, 80.2341),
    "anna nagar":      (13.0850, 80.2101),
    "velachery":       (12.9815, 80.2180),
    # Pune
    "koregaon park":   (18.5362, 73.8938),
    "hinjewadi":       (18.5912, 73.7389),
    "hadapsar":        (18.5018, 73.9252),
}

def get_coordinates(place: str, city: str) -> tuple:
    """
    Returns (lat, lon, source) or raises with a clear message.
    Tries: 1) manual lookup  2) TomTom geocode with city filter
    Never silently falls back to a wrong location.
    """
    key = place.lower().strip()
    # Remove city suffix if user typed it
    for suffix in [f", {city.lower()}", f" {city.lower()}",
                   ", india", " india", ", bengaluru", ", bangalore"]:
        key = key.replace(suffix, "").strip()

    # 1. Manual lookup for known locations
    if key in MANUAL_COORDS:
        return (*MANUAL_COORDS[key], "manual_lookup")

    # 2. TomTom geocode with city filter for accuracy
    valid, err = validate_api_key()
    if not valid:
        raise ValueError(f"API key issue: {err}")

    try:
        url = f"https://api.tomtom.com/search/2/geocode/{place}.json"
        r   = requests.get(url, params={
            "key":          TOMTOM_KEY,
            "countrySet":   "IN",       # India only
            "limit":        5,
        }, timeout=8)

        if r.status_code == 401:
            raise ValueError("TomTom API key is invalid or expired")
        if r.status_code == 403:
            raise ValueError("TomTom API key does not have access to Geocoding API")
        if r.status_code != 200:
            raise ValueError(f"TomTom geocoding returned HTTP {r.status_code}")

        data    = r.json()
        results = data.get("results", [])

        if not results:
            raise ValueError(f"Location '{place}' not found in India. Try being more specific.")

        # Filter results to prefer ones matching the selected city
        city_lower = city.lower()
        preferred  = [res for res in results
                      if city_lower in json.dumps(res).lower()]
        best = preferred[0] if preferred else results[0]

        pos = best["position"]
        return (pos["lat"], pos["lon"], "tomtom_geocode")

    except requests.Timeout:
        raise ValueError("TomTom geocoding timed out. Check your internet connection.")
    except requests.ConnectionError:
        raise ValueError("Cannot connect to TomTom API. Check your internet connection.")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Geocoding error: {e}")


def get_route(src: tuple, dst: tuple) -> dict:
    """
    Returns route data from TomTom with live traffic.
    Raises with a clear message on any failure.
    No straight-line or fake fallback.
    """
    valid, err = validate_api_key()
    if not valid:
        raise ValueError(f"API key issue: {err}")

    try:
        url = (f"https://api.tomtom.com/routing/1/calculateRoute/"
               f"{src[0]},{src[1]}:{dst[0]},{dst[1]}/json")
        r   = requests.get(url, params={
            "key":           TOMTOM_KEY,
            "traffic":       "true",
            "travelMode":    "car",
            "routeType":     "fastest",
            "computeTravelTimeFor": "all",
        }, timeout=12)

        if r.status_code == 401:
            raise ValueError("TomTom API key is invalid or expired")
        if r.status_code == 403:
            raise ValueError("TomTom API key does not have Routing API access")
        if r.status_code != 200:
            raise ValueError(f"TomTom routing returned HTTP {r.status_code}")

        data = r.json()

        if "routes" not in data or not data["routes"]:
            raise ValueError("No route found between these locations")

        return data

    except requests.Timeout:
        raise ValueError("TomTom routing timed out. Check your internet connection.")
    except requests.ConnectionError:
        raise ValueError("Cannot connect to TomTom API. Check your internet connection.")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Routing error: {e}")


def parse_route(route_data: dict) -> dict:
    """
    Parse TomTom route response into clean values.
    Returns dict with data source clearly labelled.
    Never fabricates traffic delay values.
    """
    route   = route_data["routes"][0]
    summary = route["summary"]
    points  = route["legs"][0]["points"]
    coords  = [(p["latitude"], p["longitude"]) for p in points]

    live_secs = summary.get("travelTimeInSeconds",          0)
    free_secs = summary.get("noTrafficTravelTimeInSeconds", 0)
    dist_m    = summary.get("lengthInMeters",               0)

    # Delay is ONLY computed when TomTom provides both values
    if free_secs > 0 and live_secs > 0:
        delay_secs   = max(0, live_secs - free_secs)
        has_live_traffic = True
    else:
        delay_secs   = 0
        has_live_traffic = False

    # Traffic color from real speed ratio
    if has_live_traffic and free_secs > 0:
        ratio = free_secs / live_secs
        if ratio >= 0.85:
            route_color = "green";  traffic_label = "Low Traffic"
        elif ratio >= 0.60:
            route_color = "orange"; traffic_label = "Moderate Traffic"
        else:
            route_color = "red";    traffic_label = "Heavy Traffic"
    else:
        route_color   = "#2E86AB"
        traffic_label = "No Traffic Data"

    return {
        "coords":           coords,
        "distance_km":      round(dist_m / 1000, 1),
        "travel_min":       round(live_secs / 60),
        "delay_min":        round(delay_secs / 60),
        "route_color":      route_color,
        "traffic_label":    traffic_label,
        "has_live_traffic": has_live_traffic,
        "data_source":      "TomTom Live Traffic" if has_live_traffic else "TomTom Routing (no traffic data)",
    }

# ================================================================
# LOGIN PAGE
# ================================================================
def show_login():
    st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg,#0D2137 0%,#1F4E79 60%,#2E86AB 100%) !important;
        }
        .block-container { padding-top: 2rem !important; }
        section[data-testid="stSidebar"] { display: none !important; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        div[data-testid="stForm"] {
            background: white; padding: 32px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        }
        .stTextInput > div > div > input {
            border-radius: 8px !important;
            border: 1.5px solid #D5E8F7 !important;
            padding: 10px 14px !important;
            font-size: 14px !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #1F4E79 !important;
            box-shadow: 0 0 0 2px rgba(31,78,121,0.15) !important;
        }
        .stButton > button {
            background: linear-gradient(135deg,#1F4E79,#2E86AB) !important;
            color: white !important; border-radius: 8px !important;
            border: none !important; padding: 12px !important;
            font-size: 15px !important; font-weight: bold !important;
            width: 100% !important; margin-top: 8px !important;
            box-shadow: 0 4px 12px rgba(31,78,121,0.3) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.1, 1])
    with center:
        st.markdown("""
        <div style="text-align:center;margin-bottom:24px;padding-top:16px">
            <div style="font-size:52px">🏙️</div>
            <h1 style="color:white;font-size:26px;margin:6px 0 0;font-family:Arial">
                UrbanAI Navigator
            </h1>
            <p style="color:#AED6F1;font-size:12px;margin:4px 0 0">
                Smart City Life Optimizer
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not db_exists():
            st.error("Database not found. Run: python create_db.py", icon="🗄️")

        # Tabs for Sign In / Sign Up
        sign_in_tab, sign_up_tab = st.tabs(["Sign In", "Create Account"])

        # ── SIGN IN ──────────────────────────────────────────
        with sign_in_tab:
            with st.form("login_form"):
                st.markdown("""
                <div style="text-align:center;margin-bottom:16px">
                    <h3 style="color:#1F4E79;margin:0;font-size:18px">
                        Welcome Back
                    </h3>
                </div>""", unsafe_allow_html=True)

                username  = st.text_input("Username",
                                           placeholder="Enter your username")
                password  = st.text_input("Password", type="password",
                                           placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In",
                                                   use_container_width=True)

                if submitted:
                    if not username or not password:
                        st.error("Please fill in both fields.")
                    elif not db_exists():
                        st.error("Database not found. Run: python create_db.py")
                    else:
                        user = get_user(username)
                        if not user:
                            st.error("Username not found.")
                        elif user["password"] != hash_pw(password):
                            st.error("Incorrect password.")
                        else:
                            st.session_state.update({
                                "logged_in":  True,
                                "username":   username.lower(),
                                "user_name":  user["name"],
                                "user_role":  user["role"],
                                "login_time": datetime.datetime.now(),
                            })
                            st.rerun()

        # ── SIGN UP ──────────────────────────────────────────
        with sign_up_tab:
            with st.form("signup_form"):
                st.markdown("""
                <div style="text-align:center;margin-bottom:16px">
                    <h3 style="color:#1F4E79;margin:0;font-size:18px">
                        Create Account
                    </h3>
                    <p style="color:#888;font-size:11px;margin:4px 0 0">
                        New accounts are assigned Viewer role
                    </p>
                </div>""", unsafe_allow_html=True)

                new_name  = st.text_input("Full Name",
                                           placeholder="Your full name")
                new_user  = st.text_input("Username",
                                           placeholder="Choose a username",
                                           key="signup_user")
                new_pass  = st.text_input("Password", type="password",
                                           placeholder="Min 8 characters",
                                           key="signup_pass")
                confirm   = st.text_input("Confirm Password", type="password",
                                           placeholder="Re-enter password",
                                           key="signup_confirm")
                register  = st.form_submit_button("Create Account",
                                                   use_container_width=True)

                if register:
                    if not all([new_name, new_user, new_pass, confirm]):
                        st.error("Please fill in all fields.")
                    elif len(new_user.strip()) < 3:
                        st.error("Username must be at least 3 characters.")
                    elif len(new_pass) < 8:
                        st.error("Password must be at least 8 characters.")
                    elif new_pass != confirm:
                        st.error("Passwords do not match.")
                    elif not db_exists():
                        st.error("Database not found. Run: python create_db.py")
                    elif username_exists(new_user):
                        st.error("Username already taken. Try another.")
                    else:
                        if create_user(new_user, new_pass, new_name):
                            st.success(
                                f"Account created for {new_name}! "
                                "Please sign in using the Sign In tab."
                            )
                        else:
                            st.error("Account creation failed. Please try again.")

        st.markdown("""
        <div style="text-align:center;margin-top:18px">
            <p style="color:#AED6F1;font-size:11px">
                VTU Final Year Project 2025-26 &nbsp;|&nbsp;
                GEC Karwar, Dept of CSE
            </p>
        </div>
        """, unsafe_allow_html=True)


# ================================================================
# LOGOUT
# ================================================================
def logout():
    for k in ["logged_in","username","user_name","user_role","login_time"]:
        st.session_state.pop(k, None)
    st.rerun()


# ================================================================
# GATE
# ================================================================
if not st.session_state.get("logged_in"):
    show_login()
    st.stop()


# ================================================================
# APP CSS (only shown when logged in)
# ================================================================
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
    .info-box {
        background: white; padding: 14px; border-radius: 10px;
        border-left: 4px solid #2E86AB;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06); margin: 6px 0;
    }
    .source-live {
        background: #EAFAF1; border-left: 3px solid #27AE60;
        padding: 6px 10px; border-radius: 6px;
        font-size: 12px; color: #1E8449;
    }
    .source-cached {
        background: #FEF9E7; border-left: 3px solid #F39C12;
        padding: 6px 10px; border-radius: 6px;
        font-size: 12px; color: #7D6608;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ================================================================
# CITY DATA
# ================================================================
CITY_DATA = {
    "Bangalore": {"center":[12.9716,77.5946],"zoom":12,
                  "examples":("MG Road, Bangalore","Silk Board, Bangalore")},
    "Mumbai":    {"center":[19.0760,72.8777],"zoom":12,
                  "examples":("Bandra, Mumbai","Andheri, Mumbai")},
    "Delhi":     {"center":[28.6139,77.2090],"zoom":11,
                  "examples":("Connaught Place, Delhi","Lajpat Nagar, Delhi")},
    "Hyderabad": {"center":[17.3850,78.4867],"zoom":12,
                  "examples":("Hitech City, Hyderabad","Charminar, Hyderabad")},
    "Chennai":   {"center":[13.0827,80.2707],"zoom":12,
                  "examples":("T Nagar, Chennai","Anna Nagar, Chennai")},
    "Pune":      {"center":[18.5204,73.8567],"zoom":12,
                  "examples":("Koregaon Park, Pune","Hinjewadi, Pune")},
}

# ================================================================
# MAP BUILDERS
# ================================================================
def build_default_map(city_info: dict) -> folium.Map:
    """Plain city map — no heatmap, no fake data"""
    return folium.Map(
        location=city_info["center"],
        zoom_start=city_info["zoom"],
        tiles="OpenStreetMap"
    )


def build_route_map(src_coord, dst_coord, route_coords,
                    route_color, source_name, dest_name,
                    traffic_label, has_live) -> folium.Map:
    """
    Real road route map.
    Clearly shows whether traffic data is live or unavailable.
    """
    center = [(src_coord[0]+dst_coord[0])/2,
              (src_coord[1]+dst_coord[1])/2]

    m = folium.Map(location=center, zoom_start=13,
                   tiles="OpenStreetMap")

    # White outline for depth
    folium.PolyLine(route_coords, color="white",
                    weight=12, opacity=0.6).add_to(m)

    # Colored route
    folium.PolyLine(route_coords, color=route_color,
                    weight=7, opacity=0.95,
                    tooltip=f"{source_name} → {dest_name} | {traffic_label}"
                    ).add_to(m)

    # Start marker
    folium.Marker(
        src_coord,
        tooltip=f"Start: {source_name}",
        popup=folium.Popup(f"<b>Start</b><br>{source_name}", max_width=160),
        icon=folium.Icon(color="green", icon="circle", prefix="glyphicon")
    ).add_to(m)

    # End marker
    folium.Marker(
        dst_coord,
        tooltip=f"End: {dest_name}",
        popup=folium.Popup(f"<b>Destination</b><br>{dest_name}", max_width=160),
        icon=folium.Icon(color="red", icon="map-marker", prefix="glyphicon")
    ).add_to(m)

    # Traffic label badge — shows REAL source
    mid   = route_coords[len(route_coords) // 2]
    badge = (f"🔴 {traffic_label}" if route_color=="red"
             else f"🟠 {traffic_label}" if route_color=="orange"
             else f"🟢 {traffic_label}" if route_color=="green"
             else f"ℹ️ {traffic_label}")

    folium.Marker(
        mid,
        icon=folium.DivIcon(
            html=f"""<div style="background:{'#1F4E79' if not has_live else route_color};
                color:white;padding:4px 10px;border-radius:12px;
                font-size:11px;font-weight:bold;font-family:Arial;
                white-space:nowrap;box-shadow:0 2px 4px rgba(0,0,0,0.3)">
                {badge}</div>""",
            icon_size=(160, 26), icon_anchor=(80, 13)
        )
    ).add_to(m)

    # Fit map to route
    lats = [c[0] for c in route_coords]
    lons = [c[1] for c in route_coords]
    m.fit_bounds([
        [min(lats)-0.01, min(lons)-0.01],
        [max(lats)+0.01, max(lons)+0.01]
    ])
    return m


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    # User card
    user_name  = st.session_state.get("user_name",  "User")
    user_role  = st.session_state.get("user_role",  "Viewer")
    login_time = st.session_state.get("login_time", datetime.datetime.now())

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.12);border-radius:10px;
                padding:12px;margin-bottom:8px;
                border:1px solid rgba(255,255,255,0.2)">
        <div style="font-size:26px;text-align:center">👤</div>
        <div style="color:white;font-weight:bold;text-align:center;font-size:13px">
            {user_name}
        </div>
        <div style="color:#AED6F1;text-align:center;font-size:11px">{user_role}</div>
        <div style="color:#7EC8E3;text-align:center;font-size:10px;margin-top:2px">
            Since {login_time.strftime('%H:%M')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Logout", use_container_width=True):
        logout()

    st.markdown("---")
    st.title("UrbanAI Navigator")
    st.markdown("*Smart City Life Optimizer*")
    st.markdown("---")

    city      = st.selectbox("Select City", list(CITY_DATA.keys()))
    city_info = CITY_DATA[city]
    default_src, default_dst = city_info["examples"]

    source      = st.text_input("Source",      default_src,
                                 placeholder="e.g. MG Road")
    destination = st.text_input("Destination", default_dst,
                                 placeholder="e.g. Silk Board")
    get_route_btn = st.button("Get Route", type="primary",
                               use_container_width=True)

    if st.session_state.get("route_active"):
        if st.button("Clear Route", use_container_width=True):
            for k in ["route_active","distance","travel_time",
                      "traffic_level","delay","src_coord","dst_coord",
                      "route_coords","route_color","source_name",
                      "dest_name","data_source","has_live_traffic",
                      "traffic_label"]:
                st.session_state.pop(k, None)
            st.rerun()

    # API key status
    st.markdown("---")
    key_ok, key_msg = validate_api_key()
    if key_ok:
        st.success("TomTom API: Connected")
    else:
        st.error(f"TomTom API: {key_msg}")


# ================================================================
# MAIN PAGE
# ================================================================
st.title("UrbanAI Navigator")
st.markdown(f"*Smart City Life Optimizer — {city}*")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("City",        city)
c2.metric("Distance",    st.session_state.get("distance",      "--"))
c3.metric("Travel Time", st.session_state.get("travel_time",   "--"))
c4.metric("Traffic",     st.session_state.get("traffic_level", "--"))
st.markdown("---")

# ================================================================
# ROUTE LOGIC
# ================================================================
if get_route_btn:
    if not source or not destination:
        st.warning("Please enter both source and destination.")
    else:
        key_ok, key_msg = validate_api_key()
        if not key_ok:
            st.error(f"Cannot get route: {key_msg}")
        else:
            src_coord = dst_coord = None
            route_data = None
            error_occurred = False

            # Step 1: Geocode source
            with st.spinner(f"Finding {source}..."):
                try:
                    src_coord = get_coordinates(source, city)
                except ValueError as e:
                    st.error(f"Source location error: {e}")
                    error_occurred = True

            # Step 2: Geocode destination
            if not error_occurred:
                with st.spinner(f"Finding {destination}..."):
                    try:
                        dst_coord = get_coordinates(destination, city)
                    except ValueError as e:
                        st.error(f"Destination location error: {e}")
                        error_occurred = True

            # Step 3: Get route
            if not error_occurred:
                with st.spinner("Calculating route with live traffic..."):
                    try:
                        route_data = get_route(src_coord[:2], dst_coord[:2])
                    except ValueError as e:
                        st.error(f"Routing error: {e}")
                        error_occurred = True

            # Step 4: Parse and display
            if not error_occurred and route_data:
                parsed = parse_route(route_data)

                # Save to session
                st.session_state.update({
                    "route_active":     True,
                    "src_coord":        src_coord[:2],
                    "dst_coord":        dst_coord[:2],
                    "route_coords":     parsed["coords"],
                    "route_color":      parsed["route_color"],
                    "traffic_label":    parsed["traffic_label"],
                    "has_live_traffic": parsed["has_live_traffic"],
                    "data_source":      parsed["data_source"],
                    "source_name":      source,
                    "dest_name":        destination,
                    "distance":         f"{parsed['distance_km']} km",
                    "travel_time":      f"{parsed['travel_min']} min",
                    "traffic_level":    parsed["traffic_label"].split()[0],
                    "delay":            (f"+{parsed['delay_min']} min delay"
                                         if parsed["delay_min"] > 0 else "No delay"),
                })

                st.rerun()

                # Traffic banner
                if parsed["traffic_label"] == "Heavy Traffic":
                    st.error(f"Heavy traffic — +{parsed['delay_min']} min delay. "
                             f"Try travelling after 10 AM or after 8 PM.")
                elif parsed["traffic_label"] == "Moderate Traffic":
                    st.warning(f"Moderate traffic — +{parsed['delay_min']} min delay.")
                elif parsed["traffic_label"] == "Low Traffic":
                    st.success("Clear route — great time to travel!")
                else:
                    st.info("Route calculated. No live traffic data available for this route.")

                # Data source label — always honest
                source_css = "source-live" if parsed["has_live_traffic"] else "source-cached"
                st.markdown(
                    f'<div class="{source_css}">Data source: {parsed["data_source"]}</div>',
                    unsafe_allow_html=True
                )

# ================================================================
# MAP
# ================================================================
if st.session_state.get("route_active"):
    city_map = build_route_map(
        st.session_state["src_coord"],
        st.session_state["dst_coord"],
        st.session_state["route_coords"],
        st.session_state["route_color"],
        st.session_state["source_name"],
        st.session_state["dest_name"],
        st.session_state["traffic_label"],
        st.session_state["has_live_traffic"],
    )
else:
    city_map = build_default_map(city_info)

st_folium(city_map, width=1100, height=560, returned_objects=[])

# Route details below map
if st.session_state.get("route_active"):
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f'<div class="info-box"><b>From</b><br>{st.session_state.get("source_name","")}</div>',
        unsafe_allow_html=True)
    col2.markdown(
        f'<div class="info-box"><b>To</b><br>{st.session_state.get("dest_name","")}</div>',
        unsafe_allow_html=True)
    col3.markdown(
        f'<div class="info-box"><b>Delay</b><br>{st.session_state.get("delay","--")}</div>',
        unsafe_allow_html=True)
    col4.markdown(
        f'<div class="info-box"><b>Data Source</b><br>'
        f'{st.session_state.get("data_source","--")}</div>',
        unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<center><small style='color:#888'>UrbanAI Navigator | "
    "VTU Final Year Project 2025-26 | GEC Karwar, Dept of CSE</small></center>",
    unsafe_allow_html=True
)

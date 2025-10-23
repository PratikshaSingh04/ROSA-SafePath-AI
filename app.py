import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
from geopy.geocoders import Nominatim
import geocoder
import pandas as pd
import os
from datetime import datetime
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString
import random
import numpy as np 
from folium import features as f
import time

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(page_title="ROSA – SafePath AI", layout="wide")

# ------------------------------------------------------------
# SESSION STATE INITIALIZATION (For User Reports)
# ------------------------------------------------------------
if 'user_reports' not in st.session_state:
    st.session_state.user_reports = pd.DataFrame(columns=['lat', 'lon', 'hazard', 'timestamp'])

# ------------------------------------------------------------
# CUSTOM PAGE STYLE (Interactive & Polished Aesthetics)
# ------------------------------------------------------------
COLOR_PRIMARY = "#C2185B"    # ROSA Pink
COLOR_BACKGROUND = "#212121" # Deep Charcoal
COLOR_SECONDARY_BG = "#333333" # Dark Gray

page_bg = f"""
<style>
/* 1. Global & Backgrounds */
[data-testid="stAppViewContainer"] {{ background-color: {COLOR_BACKGROUND}; }}
[data-testid="stHeader"] {{ background-color: rgba(0,0,0,0); }}

/* 2. Headers, Branding, and Sidebar */
h1, h2, h3, h4, h5, h6 {{ color: {COLOR_PRIMARY} !important; font-family: 'Segoe UI', sans-serif; }}
[data-testid="stSidebar"] {{ background-color: {COLOR_SECONDARY_BG}; border-right: 1px solid #444; }}

/* 3. General Widget and Input Hover Effect (NEW) */
div.stSelectbox, div.stTextInput, div.stDateInput {{
    border-radius: 8px;
    transition: box-shadow 0.3s ease-in-out;
}}
div.stSelectbox:hover, div.stTextInput:hover {{
    box-shadow: 0 0 10px rgba(194, 24, 91, 0.5); /* Pink Shadow/Glow on hover */
}}

/* 4. Button Styling (General) */
.stButton>button {{ 
    background-color: {COLOR_PRIMARY}; 
    color: white; 
    border-radius: 8px; 
    padding: 0.5em 1em; 
    border: none; 
    transition: background-color 0.2s, box-shadow 0.3s;
}}
.stButton>button:hover {{ 
    background-color: #9c1549; 
    box-shadow: 0 4px 10px rgba(194, 24, 91, 0.4); /* Lift and Shadow */
}}

/* 5. Map Container Shadow (NEW) */
[data-testid="stDeck"] {{
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5); 
}}

/* 6. SOS Button Glow (NEW) */
.sos-button > button {{
    box-shadow: 0 0 15px rgba(255, 0, 0, 0.7);
    animation: pulse 1.5s infinite;
}}
@keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }}
    70% {{ box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }}
}}

/* 7. Info/Warning Box Lift Effect (NEW) */
.stAlert {{ 
    border-radius: 10px; 
    border-left: 5px solid {COLOR_PRIMARY}; 
    transition: transform 0.3s ease-in-out, box-shadow 0.3s;
}}
.stAlert:hover {{
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}}

/* 8. Metric Styling */
[data-testid="stMetricValue"] {{
    font-size: 2.5rem !important;
    font-weight: 700 !important;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD OR GENERATE SAFETY DATA 
# ------------------------------------------------------------
@st.cache_data 
def load_safety_data():
    path = "safety_data.csv"
    if not os.path.exists(path):
        areas = [
             ("Delhi", "Connaught Place", 28.6315, 77.2167, "metro_station"), 
             ("Delhi", "India Gate", 28.6129, 77.2295, "isolated_park"),
             ("Delhi", "Saket Malls", 28.5245, 77.2060, "nightclub/bar"),
             ("Delhi", "Dwarka Sector 12", 28.5876, 77.0460, "police_station"),
             ("Delhi", "Lajpat Nagar Market", 28.5686, 77.2439, "metro_station"),
             ("Delhi", "Hauz Khas Village", 28.5494, 77.2001, "nightclub/bar"),
             ("Delhi", "Rajouri Garden", 28.6422, 77.1174, "metro_station"),
             
             ("Noida", "Noida Sector 18", 28.5707, 77.3260, "nightclub/bar"), 
             ("Noida", "Noida Sector 62", 28.6304, 77.3733, "police_station"),
             
             ("Gurgaon", "Gurgaon Cyber Hub", 28.4945, 77.0880, "nightclub/bar"), 
             ("Gurgaon", "DLF Phase 3", 28.4839, 77.1015, "metro_station"),
             
             ("Greater Noida", "Pari Chowk", 28.4744, 77.5030, "metro_station"), 
             ("Greater Noida", "Alpha 1 Park", 28.4702, 77.5095, "isolated_park"),
             
             ("Ghaziabad", "Ghaziabad Raj Nagar", 28.6670, 77.4440, "police_station"),
             ("Ghaziabad", "Indirapuram", 28.6437, 77.3727, "metro_station"),
             ("Ghaziabad", "Kaushambi Bus Stop", 28.6430, 77.3283, "isolated_park"),
        ]
        data = []
        
        CONTEXT_SCORES = {
            "police_station":      {"reports": (1, 5),   "lighting": (4.5, 5.0), "cctv": (4.0, 5.0), "crowd": (3.5, 4.5), "rating": (4.0, 5.0)},
            "metro_station":       {"reports": (5, 12),  "lighting": (3.5, 4.5), "cctv": (3.0, 4.5), "crowd": (4.0, 5.0), "rating": (3.5, 4.5)},
            "nightclub/bar":       {"reports": (15, 30), "lighting": (2.5, 4.0), "cctv": (1.0, 3.5), "crowd": (3.0, 4.0), "rating": (2.5, 3.5)},
            "isolated_park":       {"reports": (10, 25), "lighting": (1.0, 3.0), "cctv": (1.0, 2.5), "crowd": (1.0, 2.5), "rating": (2.5, 3.5)},
        }

        for city, name, lat, lon, poi_type in areas:
            scores = CONTEXT_SCORES.get(poi_type, CONTEXT_SCORES["metro_station"])

            data.append({
                "area": name,
                "city": city,
                "latitude": lat,
                "longitude": lon,
                "type": poi_type, 
                "reports": random.randint(*scores["reports"]), 
                "lighting": round(random.uniform(*scores["lighting"]), 1),
                "cctv": round(random.uniform(*scores["cctv"]), 1),
                "crowd_density": round(random.uniform(*scores["crowd"]), 1),
                "user_rating": round(random.uniform(*scores["rating"]), 1)
            })
        pd.DataFrame(data).to_csv(path, index=False)
    
    df = pd.read_csv(path)
    if 'type' not in df.columns:
        st.error("Error: Data file is outdated. Please delete 'safety_data.csv' and restart.")
        st.stop()
        
    return df

safety_data = load_safety_data()

# ------------------------------------------------------------
# CORE AI AND HELPER FUNCTIONS 
# ------------------------------------------------------------

def get_coordinates(place):
    global city_choice 
    geolocator = Nominatim(user_agent="safe_path_ai")
    try:
        location = geolocator.geocode(f"{place}, {city_choice}", timeout=10) 
        return (location.latitude, location.longitude) if location else None
    except:
        return None

def get_current_location():
    g = geocoder.ip('me')
    return tuple(g.latlng) if g.latlng else None

def nearest_area(lat, lon):
    data_copy = safety_data.copy()
    data_copy["dist"] = ((data_copy["latitude"] - lat)**2 + (data_copy["longitude"] - lon)**2)**0.5
    
    if not st.session_state.user_reports.empty:
        matched_area = data_copy.sort_values("dist").iloc[0]
        report_dist_threshold = 0.01 
        
        for index, report in st.session_state.user_reports.iterrows():
            if abs(report['lat'] - matched_area['latitude']) < report_dist_threshold and \
               abs(report['lon'] - matched_area['longitude']) < report_dist_threshold:
                matched_area['reports'] += 5
        
        return matched_area 

    return data_copy.sort_values("dist").iloc[0]

def calculate_route_safety_score(route_coords):
    if not route_coords:
        return 50 
        
    hour = datetime.now().hour
    time_factor = 1.0 if (hour > 5 and hour < 20) else 0.75 

    if hour >= 0 and hour <= 5:
        crowd_sim_factor = random.uniform(-1.0, 0.5) 
        crowd_weight_mod = 0.8 
    elif hour >= 17 and hour <= 23:
        crowd_sim_factor = random.uniform(0.5, 1.5) 
        crowd_weight_mod = 1.2 
    else:
        crowd_sim_factor = random.uniform(0.0, 0.5)
        crowd_weight_mod = 1.0

    sample_size = max(1, len(route_coords) // 25)
    sampled_points = route_coords[::sample_size]
    
    safety_metrics = []
    
    for lat, lon in sampled_points:
        nearest = nearest_area(lat, lon) 
        
        current_crowd_score = max(1.0, nearest["crowd_density"] + crowd_sim_factor)

        safety_metric_base = (
            nearest["lighting"]*0.2 + 
            nearest["cctv"]*0.2 + 
            current_crowd_score*0.1*crowd_weight_mod +
            nearest["user_rating"]*0.3
        ) 
        
        report_penalty = nearest["reports"] * 0.1 
        
        point_metric = max(0, safety_metric_base - report_penalty)
        safety_metrics.append(point_metric)
            
    min_metric = min(safety_metrics) if safety_metrics else 5
    
    final_score = int(min_metric * 20 * time_factor) 
    return min(100, max(30, final_score))

def get_safe_route(G, orig_node, dest_node):
    
    RISK_FACTOR = 5000 
    
    for u, v, k, data in G.edges(keys=True, data=True):
        try:
            if 'geometry' in data:
                line = data['geometry']
                mid_lat, mid_lon = line.centroid.y, line.centroid.x
            else:
                mid_lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
                mid_lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
        except:
            mid_lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
            mid_lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2

        nearest = nearest_area(mid_lat, mid_lon)
        
        safety_penalty = (5 - nearest["user_rating"]) * 5 + nearest["reports"] * 0.5 
        
        data['safety_cost'] = data.get('length', 1) + (safety_penalty * RISK_FACTOR)

    return nx.shortest_path(G, orig_node, dest_node, weight='safety_cost')

@st.cache_data(show_spinner="Calculating shortest and safest routes...")
def get_routes(start_coords, end_coords):
    try:
        for radius in [3000, 8000, 15000, 25000]:
            try:
                G = ox.graph_from_point(start_coords, dist=radius, network_type='drive')
                orig_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
                dest_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
                
                shortest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight='length')
                shortest_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in shortest_route_nodes]
                
                safest_route_nodes = get_safe_route(G, orig_node, dest_node)
                safest_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in safest_route_nodes]
                
                if safest_coords[-1] != end_coords:
                     safest_coords.append(end_coords)
                if shortest_coords[-1] != end_coords:
                     shortest_coords.append(end_coords)

                shortest_score = calculate_route_safety_score(shortest_coords)
                safest_score = calculate_route_safety_score(safest_coords)
                
                return shortest_coords, safest_coords, shortest_score, safest_score
            except nx.NetworkXNoPath:
                continue
        
        st.warning("Could not find a connecting route in the area. Displaying straight line.")
        return [start_coords, end_coords], [start_coords, end_coords], 50, 50
    except Exception as e:
        st.error(f"Route calculation failed: {e}")
        return [start_coords, end_coords], [start_coords, end_coords], 50, 50

# ------------------------------------------------------------
# SOS LOGGING & UI HELPERS
# ------------------------------------------------------------
SOS_FILE = "sos_log.csv"

def save_sos_alert(name, number, location_text, coords):
    data = {
        "Timestamp": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        "Name": [name],
        "Number": [number],
        "Location": [location_text],
        "Coordinates": [coords]
    }
    df = pd.DataFrame(data)
    if os.path.exists(SOS_FILE):
        df.to_csv(SOS_FILE, mode='a', index=False, header=False)
    else:
        df.to_csv(SOS_FILE, index=False)
        
# Function to display colored metric (Polished)
def display_colored_metric(col, title, score, help_text):
    """Displays st.metric with dynamically colored value based on safety score."""
    # Define colors based on score tiers
    if score >= 85: color = "#4CAF50" # Green
    elif score >= 70: color = "#FFC107" # Yellow
    else: color = "#C2185B" # Red
    
    # Inject CSS to override the st.metric value color
    st.markdown(
        f"""
        <style>
        [data-testid="stMetricValue"] {{
            color: {color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    col.metric(title, f"{score}/100", delta_color="off", help=help_text)

def simulate_safe_journey(shortest_coords, safest_coords, shortest_score, safest_score):
    st.subheader("Journey Mode: ROSA is Guiding You")
    
    coords_to_follow = safest_coords if safest_score >= shortest_score else shortest_coords
    total_steps = 100 
    
    progress_bar = st.progress(0, text="Starting journey...")
    current_location_text = st.empty()
    check_in_button_container = st.empty()

    for i in range(total_steps + 1):
        if i == 50:
            with check_in_button_container:
                if st.button("Safety Check-in: Tap to confirm you are safe", type="primary", key="check_in_btn"):
                    st.toast("Safety confirmed. Continuing route.")
                    check_in_button_container.empty()
                    
        progress_bar.progress(i, text=f"Progress: {i}% complete.")
        
        if i % 10 == 0 and i != 0:
            index = int((i/100) * (len(coords_to_follow) - 1))
            lat, lon = coords_to_follow[index]
            nearest = nearest_area(lat, lon) 
            current_location_text.info(f"Location Estimate (Simulated): Nearing **{nearest['area']}**")
        
        time.sleep(0.1) 

    progress_bar.empty()
    current_location_text.success("Journey Complete! You have arrived safely.")


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
# Assuming your logo file is named 'rosa_logo.png' and is in the same directory as app.py
logo_path = "rosa_logo.png"

# Display the logo and title
st.image(logo_path, width=250) 
st.markdown("Empowering individuals through AI-driven safe travel insights and instant SOS assistance.")
import pytz # Import the pytz library

# --- Header Section ---
# ... (Logo and tagline code) ...

# 1. Get the current time in UTC
utc_now = datetime.now(pytz.utc)

# 2. Define the Indian Standard Time zone
ist_timezone = pytz.timezone('Asia/Kolkata')

# 3. Localize the UTC time to IST
ist_now = utc_now.astimezone(ist_timezone)

# 4. Display the localized time (IST)
st.write(f"Current Time (IST): **{ist_now.strftime('%A, %d %B %Y | %I:%M %p')}**")
# ------------------------------------------------------------
# SIDEBAR INPUTS 
# ------------------------------------------------------------
st.sidebar.header("Route Details & City Selection")

# City selector
city_choice = st.sidebar.selectbox("Select City", sorted(safety_data["city"].unique()))

# Filter areas for the selected city
city_areas = safety_data[safety_data["city"] == city_choice]["area"].tolist()

start_location, end_location = None, None
start_coords, end_coords = None, None 

if not city_areas:
    st.sidebar.warning("No areas found for this city. Please check your dataset.")
else:
    # Use columns in sidebar for a cleaner, packed look 
    col_start, col_end = st.sidebar.columns(2) 
    
    start_index = 0
    end_index = 1 if len(city_areas) > 1 else 0

    start_location = col_start.selectbox("Starting Point", city_areas, index=start_index)
    end_location = col_end.selectbox("Destination", city_areas, index=end_index)
    
    if start_location == end_location and len(city_areas) > 1:
        st.sidebar.error("Start and Destination must be different!")
        start_location = None
        end_location = None
    
    if start_location and end_location:
        start_coords = get_coordinates(start_location)
        end_coords = get_coordinates(end_location)


st.sidebar.markdown("---")
st.sidebar.header("Emergency Contact Setup")
contact_name = st.sidebar.text_input("Contact Name", "Mom", help="Used for the WhatsApp link message.")
contact_number = st.sidebar.text_input("Contact Number (e.g., +91XXXXXXXXXX)", "+91XXXXXXXXXX", help="Your trusted contact's number, used for the WhatsApp SOS link.")


# ------------------------------------------------------------
# MAIN MAP + SAFETY SECTION
# ------------------------------------------------------------
if start_coords and end_coords:
    
    shortest_coords, safest_coords, shortest_score, safest_score = get_routes(start_coords, end_coords)

    # Map setup
    all_coords = shortest_coords + safest_coords
    lat_min = min(c[0] for c in all_coords)
    lat_max = max(c[0] for c in all_coords)
    lon_min = min(c[1] for c in all_coords)
    lon_max = max(c[1] for c in all_coords)
    
    m = folium.Map(location=start_coords, zoom_start=12, tiles="cartodbdarkmatter")
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]]) 

    # 2. Visualize Routes
    folium.PolyLine(
        safest_coords, 
        color="#C2185B", 
        weight=6, 
        opacity=1.0, 
        tooltip=f"Safest Route Score: {safest_score}/100"
    ).add_to(m)
    
    if shortest_coords != safest_coords or safest_score < 90: 
        folium.PolyLine(
            shortest_coords, 
            color="#888888", # Lighter gray for better visibility
            weight=5, # Increased weight
            opacity=0.8, 
            tooltip=f"Shortest Route Score: {shortest_score}/100",
            dash_array='8, 8' 
        ).add_to(m)

    # Markers
    folium.Marker(start_coords, tooltip="Start", icon=folium.Icon(color="green", icon="fa-person-walking", prefix="fa")).add_to(m)
    folium.Marker(end_coords, tooltip="Destination", icon=folium.Icon(color="darkred", icon="fa-flag", prefix="fa")).add_to(m)

    # 3. Safety Data Visualization 
    heat_data = [[row['latitude'], row['longitude'], row['reports'] * row['crowd_density'] / 5] for _, row in safety_data.iterrows()]
    HeatMap(heat_data, radius=15, blur=10, max_zoom=12, name="Risk Heatmap").add_to(m)
    
    poi_cluster = MarkerCluster(name="Key POIs (Safety/Risk)").add_to(m)

    for _, row in safety_data.iterrows():
        
        # Hazard Marker Highlight (Visible on dark map)
        if row['reports'] > 15 and row['type'] in ['nightclub/bar', 'isolated_park']:
             folium.Marker(
                location=[row['latitude'], row['longitude']],
                tooltip=f"HIGH RISK: {row['area']} | Reports: {row['reports']}",
                icon=folium.Icon(color='orange', icon='fa-exclamation-triangle', prefix='fa') # Orange marker for visibility
            ).add_to(poi_cluster)
        else:
            # Standard POI Marker
            icon_color = "darkgreen" if row['type'] == 'police_station' else ("blue" if row['type'] == 'metro_station' else ("darkred" if row['type'] == 'nightclub/bar' else "orange"))
            icon_name = "fa-shield-halved" if row['type'] == 'police_station' else ("fa-train-subway" if row['type'] == 'metro_station' else ("fa-bell-slash" if row['type'] == 'isolated_park' else "fa-martini-glass-citrus"))
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                tooltip=f"{row['type'].title()}: {row['area']}<br>Reports: {row['reports']}",
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
            ).add_to(poi_cluster)

    folium.LayerControl().add_to(m)

    st_folium(m, width=800, height=500)
    
    # 4. Score Comparison Display 
    st.markdown("---")
    st.subheader("Route Analysis & Recommendation")
    
    # Display Dynamic Factors (Text only)
    hour = datetime.now().hour
    if hour >= 0 and hour <= 5:
        time_desc = "Night (00:00 - 05:00)"
        risk_adj = "Increased risk due to low crowd density."
    elif hour >= 17 and hour <= 23:
        time_desc = "Peak Evening (17:00 - 23:00)"
        risk_adj = "Crowd factor adjusted to high activity levels."
    else:
        time_desc = "Daytime (06:00 - 16:00)"
        risk_adj = "Standard daylight assessment."

    st.markdown(f"**Current Context:** **{time_desc}** | *{risk_adj}*")
    
    avg_city_rating = safety_data[safety_data['city'] == city_choice]['user_rating'].mean()
    st.info(f"City Average Safety Rating ({city_choice}): **{avg_city_rating:.1f}/5**")

    col_shortest, col_safest, col_advice = st.columns([1, 1, 2])
    
    # Use custom function for colored metrics 
    display_colored_metric(col_shortest, "Shortest Path Score (Grey Line)", shortest_score, "Score based on weighted averages of features along the route.")
    display_colored_metric(col_safest, "Safest Path Score (Red Line)", safest_score, "Score based on weighted averages of features along the route.")

    # 5. Final Recommendation and Advice
    with col_advice:
        st.markdown(f"#### Overall Safety Advice (Time-Aware)")
        
        if safest_score > shortest_score + 5: 
            st.success("Recommendation: Take the Safest Route (Red Line). It avoids higher-risk zones and scores significantly better.")
        elif shortest_score >= 80 and safest_score < shortest_score + 5:
            st.info("Recommendation: Either route is good. Both are rated relatively safe. You can choose the shorter path (Grey Line).")
        else:
            st.warning("Caution: Both routes have areas of concern, especially given the current time.")

    # ------------------------------------------------------------
    # USER FEEDBACK / REPORTING 
    # ------------------------------------------------------------
    with st.expander("Report a Local Hazard (User Feedback System)"):
        report_cols = st.columns([1, 1, 1])
        hazard = report_cols[0].selectbox("Hazard Type", ["Street Light Out", "Suspicious Activity", "Road Blockage", "Other Safety Concern"])
        
        if report_cols[1].button("Report Hazard at Start Location", key="report_start"):
            if start_coords:
                new_report = pd.DataFrame([{'lat': start_coords[0], 'lon': start_coords[1], 'hazard': hazard, 'timestamp': datetime.now()}])
                st.session_state.user_reports = pd.concat([st.session_state.user_reports, new_report], ignore_index=True)
                st.success(f"Report logged near Start Location. Recalculate route to see safety score changes!")
            else:
                st.error("Please select a valid start location first.")

        if report_cols[2].button("Report Hazard at Destination", key="report_end"):
            if end_coords:
                new_report = pd.DataFrame([{'lat': end_coords[0], 'lon': end_coords[1], 'hazard': hazard, 'timestamp': datetime.now()}])
                st.session_state.user_reports = pd.concat([st.session_state.user_reports, new_report], ignore_index=True)
                st.success(f"Report logged near Destination. Recalculate route to see safety score changes!")
            else:
                st.error("Please select a valid destination first.")

        # Recalculate Button
        if not st.session_state.user_reports.empty:
            st.markdown("---")
            st.warning("New user reports detected. Rerun analysis to see score changes.")
            if st.button("Rerun Route Analysis", key="rerun_analysis"):
                st.experimental_rerun() 


    st.markdown("---")
    # ------------------------------------------------------------
    # SOS SECTION 
    # ------------------------------------------------------------
    
    if st.button("Start Safe Journey Simulation", help="Run the journey with timed check-ins."):
        simulate_safe_journey(shortest_coords, safest_coords, shortest_score, safest_score)
        st.markdown("---") 
    
    st.subheader("Emergency Assistance (SOS)")
    
    col_sos, col_call = st.columns([1, 1])

    with col_sos:
        # Added custom CSS class for SOS button glow
        st.markdown('<div class="sos-button">', unsafe_allow_html=True)
        if st.button("Send Instant SOS Alert Now", help="Sends a pre-filled WhatsApp message to your trusted contact and logs the alert."):
            current_coords = get_current_location()
            current_location = f"Lat: {current_coords[0]:.4f}, Lon: {current_coords[1]:.4f}" if current_coords else "Unknown Location (IP estimate)"
            
            save_sos_alert(contact_name, contact_number, "Current Location", current_coords)
            
            whatsapp_msg = f"EMERGENCY! {contact_name}, I need help immediately. My current location is: {current_location}. Please track me!"
            whatsapp_link = f"https://wa.me/{contact_number.replace('+','').replace(' ','')}?text={whatsapp_msg.replace(' ','%20').replace(':','%3A').replace(',','%2C')}"
            
            st.link_button(
                f"1. Open WhatsApp to Notify {contact_name}", 
                whatsapp_link,
                type="primary",
                help="Click to open WhatsApp and send the pre-filled alert."
            )
            st.info("Alert logged successfully.")
        st.markdown('</div>', unsafe_allow_html=True) # Closing the custom div
    
    with col_call:
        st.markdown("##### Direct Emergency Call:")
        st.link_button(
            "Call National Emergency (112)", 
            "tel:112", 
            type="secondary", 
            help="Dial 112 for all emergencies in India."
        )
    
    st.markdown("---")
    st.markdown("##### Voice SOS Simulation")
    voice_trigger = st.text_input("Say 'EMERGENCY ROSA' (Type the phrase and hit Enter)", key="voice_sos")

    if voice_trigger.strip().upper() == "EMERGENCY ROSA":
        st.error("Voice Command Detected! Initiating SOS Protocol...")
        current_coords = get_current_location()
        save_sos_alert(contact_name, contact_number, "Voice Triggered SOS", current_coords)
        whatsapp_msg = f"EMERGENCY! {contact_name}, Voice SOS Triggered! My location: {current_coords}. Track me!"
        whatsapp_link = f"https://wa.me/{contact_number.replace('+','').replace(' ','')}?text={whatsapp_msg.replace(' ','%20').replace(':','%3A').replace(',','%2C')}"
        st.link_button("1. Open WhatsApp (Voice Trigger)", whatsapp_link, type="primary")


else:
    st.info("Enter both start and destination to view the route and safety score.")

# ------------------------------------------------------------
# SOS LOG DISPLAY
# ------------------------------------------------------------
if os.path.exists(SOS_FILE):
    st.markdown("#### Recent SOS Logs")
    st.dataframe(pd.read_csv(SOS_FILE).tail(5), hide_index=True)

# ------------------------------------------------------------
# ADDITIONAL SECTIONS
# ------------------------------------------------------------
st.markdown("---")

with st.expander("About ROSA & Upcoming Features"):
    st.write("""
**ROSA (Real-time Optimal Safety Assistant)** helps identify the **Safest Route** by calculating a comprehensive safety score for every segment of the journey.

**Key Features (Student Project):**
* **Contextual Data:** Safety scores are influenced by Point-of-Interest (POI) type (e.g., Police Station vs. Isolated Park).
* **Dynamic Scoring:** Safety scores change based on the current **time of day**, simulating live crowd data.
* **User Feedback Loop:** User-submitted hazard reports immediately penalize nearby route segments.
* **Simulated Tracking:** Journey mode with timed safety check-ins.
* **Dual Routing:** Compares shortest path vs. Safest Path.
""")

with st.expander("City Safety Insights"):
    st.write("**Area Safety Details** (Filtered by POI Type)")
    
    df_insights = safety_data[["city", "area", "type", "user_rating", "reports", "lighting", "cctv"]].sort_values("user_rating", ascending=False)

    def color_reports(val):
        color = 'background-color: #491515; color: white' if val > 20 else ('background-color: #8c2a2a; color: white' if val > 10 else '')
        return color
    
    st.dataframe(df_insights.style.map(color_reports, subset=['reports']), hide_index=True)
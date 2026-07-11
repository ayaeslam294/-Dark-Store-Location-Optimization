from __future__ import annotations

import base64
import os
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from pyproj import CRS
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from dark_store_ga_deap_optimizer import (
    _project_back_to_wgs84,
    build_parser,
    run_optimizer,
)

st.set_page_config(
    page_title="Dark Stores Optimizer", 
    page_icon="dark_store_logo.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

st.markdown(
    """
    <style>
        /* Hide default Streamlit header overlap */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            position: absolute !important; 
        }
        
        /* Adjust top padding */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }

        [data-testid="stHeaderActionElements"] { display: flex; }
        footer { visibility: hidden; }
        
        /* Main Mint Green Background with subtle texture/gradient */
        .stApp {
            background: linear-gradient(135deg, #6bb099 0%, #5a9b85 100%); 
            color: #ffffff;
            font-family: 'Arial Rounded MT Bold', 'Inter', sans-serif;
        }

        /* ----------------------------------------------------
       /* ----------------------------------------------------
          /* ----------------------------------------------------
           STYLISH TITLES (Glassmorphic Pills - Main Content Only)
           ---------------------------------------------------- */
        [data-testid="stMainBlockContainer"] .stMarkdown h3 {
            color: #ffffff !important;
            background: rgba(255, 255, 255, 0.2);
            /* Adjusted padding and normalized line-height */
            padding: 0.5rem 1.8rem; 
            line-height: 1.5;
            border-radius: 50px;
            /* Flexbox ensures perfect vertical centering of emojis and text */
            display: inline-flex; 
            align-items: center; 
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.4);
            margin-bottom: 1.5rem;
            font-weight: 800;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.15);
        }

        /* Streamlit sometimes adds a hidden anchor link icon inside headers; 
           this ensures it doesn't mess up our flexbox centering */
        [data-testid="stMainBlockContainer"] .stMarkdown h3 a {
            display: none !important;
        }

        /* Keep the sidebar titles clean and flat */
        [data-testid="stSidebar"] .stMarkdown h3 {
            background: transparent !important;
            color: #334155 !important;
            padding: 0;
            line-height: 1.2;
            display: block; /* Overrides the flexbox for the sidebar */
            box-shadow: none;
            border: none;
            text-shadow: none;
            margin-bottom: 1rem;
            font-weight: 800;
        }

        /* ----------------------------------------------------
           MAP & DATAFRAME FRAMING (Deep Shadows)
           ---------------------------------------------------- */
        iframe[title="streamlit_folium.st_folium"] {
            border-radius: 20px !important;
            box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.35), 0 0 20px rgba(255, 255, 255, 0.15) !important;
            border: 6px solid #ffffff !important;
            background-color: #ffffff;
        }

        [data-testid="stDataFrame"] > div {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 8px;
            box-shadow: 0 15px 30px -10px rgba(0, 0, 0, 0.2);
            border: 2px solid rgba(255, 255, 255, 0.5);
        }

        /* ----------------------------------------------------
           FLOATING CARDS & METRICS
           ---------------------------------------------------- */
        .header-card {
            background: linear-gradient(145deg, #ffffff, #f8fcfb);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.25);
            margin-bottom: 3rem;
            display: flex;
            align-items: center;
            gap: 3rem;
            flex-wrap: wrap;
            border: 1px solid rgba(255, 255, 255, 0.8);
        }
        
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #ffffff, #fdfdfd);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.9);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-8px);
            box-shadow: 0 25px 40px -15px rgba(0, 0, 0, 0.3);
        }
        [data-testid="stMetricValue"] {
            color: #f6c831 !important; 
            font-weight: 900;
            font-size: 2.4rem;
            text-shadow: 1px 1px 0px rgba(246, 200, 49, 0.2);
        }
        [data-testid="stMetricLabel"] {
            color: #64748b !important; 
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ----------------------------------------------------
           SIDEBAR & BUTTONS
           ---------------------------------------------------- */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: none;
            box-shadow: 4px 0 25px rgba(0, 0, 0, 0.1);
        }
        [data-testid="stSidebar"] * {
            color: #334155 !important; 
        }

        /* Yellow Button */
        div.stButton > button:first-child {
            background: linear-gradient(145deg, #fce054, #f6c831) !important;
            color: #1e293b !important;
            border: none;
            border-radius: 12px;
            font-weight: 800;
            box-shadow: 0 8px 15px rgba(246, 200, 49, 0.3);
            transition: all 0.2s;
            padding: 0.5rem 1rem;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(145deg, #f6c831, #eab308) !important;
            transform: translateY(-3px);
            box-shadow: 0 12px 20px rgba(246, 200, 49, 0.4);
        }

        /* Download Button */
        [data-testid="stDownloadButton"] > button {
            background-color: #ffffff !important;
            color: #6bb099 !important;
            border: 2px solid #ffffff !important;
            border-radius: 12px;
            font-weight: 800;
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
            transition: all 0.2s;
            width: 100%;
        }
        [data-testid="stDownloadButton"] > button:hover {
            color: #ffffff !important;
            background-color: #f6c831 !important;
            border-color: #f6c831 !important;
            transform: translateY(-3px);
            box-shadow: 0 12px 20px rgba(246, 200, 49, 0.3);
        }

        /* ----------------------------------------------------
           SLIDER OVERRIDE (Forcing out the default Red)
           ---------------------------------------------------- */
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: #f6c831 !important;
            border: 3px solid #ffffff !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
        }
        .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] ~ div {
            background-color: #f6c831 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

img_b64 = get_base64_image("dark_store_logo.png")
img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width: 100%; border-radius: 12px; transform: scale(1.05); filter: drop-shadow(0 10px 15px rgba(0,0,0,0.1));">' if img_b64 else '<p style="color:red;">⚠️ Image dark_store_logo.png not found</p>'

st.markdown(f"""
    <div class="header-card">
        <div style="flex: 1.2; min-width: 250px;">
            {img_tag}
        </div>
        <div style="flex: 2.5; min-width: 300px;">
            <h1 style='color: #5a9b85; margin-top: 0; padding-top: 0; font-size: 2.8rem; font-weight: 900; letter-spacing: -1px;'>Dark Store Location Scout</h1>
            <p style='color: #475569; font-size: 1.15rem; line-height: 1.7; margin-bottom: 0;'>
This data-driven spatial intelligence platform serves the rapid-delivery market expansion by identifying highly profitable dark store locations. Engineered to optimize capital allocation, it mathematically balances target market capture, supply chain efficiency, and competitive cannibalization risk.            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ Optimization Constraints")
with st.sidebar.expander("🎯 KPI Fitness Weights", expanded=True):
    coverage_weight = st.slider("Coverage Weight", min_value=0, max_value=100, value=55)
    time_weight = st.slider("Time Weight", min_value=0, max_value=100, value=35)
    competitor_weight = st.slider("Competitor Penalty Weight", min_value=0, max_value=100, value=10)

st.sidebar.markdown("---")
run_clicked = st.sidebar.button("🚀 Find Locations", type="primary", use_container_width=True)

if "solution_results" not in st.session_state:
    st.session_state.solution_results = None

def _project_results_to_wgs84(results: pd.DataFrame, target_crs: CRS) -> pd.DataFrame:
    projected_rows = []
    for _, row in results.iterrows():
        lon, lat = _project_back_to_wgs84(float(row["x"]), float(row["y"]), target_crs)
        projected_rows.append({
            "Rank": int(row["rank"]),
            "Fitness": float(row["fitness"]),
            "Average Delivery Time (min)": float(row["avg_delivery_time_min"]),
            "Population Coverage": float(row["coverage_population"]),
            "Competitor Penalty": float(row["competitor_penalty"]),
            "Longitude": float(lon),
            "Latitude": float(lat),
        })
    return pd.DataFrame(projected_rows)

@st.cache_data(show_spinner=False)
def fetch_addresses(df: pd.DataFrame) -> list[str]:
    geolocator = Nominatim(user_agent="dark_store_scout_dashboard")
    reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1.0)
    addresses = []
    for _, row in df.iterrows():
        try:
            location = reverse_geocode((row['Latitude'], row['Longitude']), language='en')
            addresses.append(location.address if location else "Unknown Location")
        except Exception:
            addresses.append("Address Offline")
    return addresses

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')

if run_clicked:
    args = build_parser().parse_args([])
    args.base_dir = Path(__file__).resolve().parent
    args.coverage_weight = float(coverage_weight)
    args.time_weight = float(time_weight)
    args.competitor_weight = float(competitor_weight)

    try:
        with st.spinner("Processing Data..."):
            results = run_optimizer(args)

        projected = _project_results_to_wgs84(results, CRS.from_user_input(args.target_crs))
        
        with st.spinner("Fetching Street Addresses..."):
            projected["Exact Address"] = fetch_addresses(projected)
            
        st.session_state.solution_results = projected
    except Exception as exc: 
        st.error(f"Optimization computation error: {exc}")

results_df = st.session_state.solution_results

if results_df is not None and not results_df.empty:
    top_solution = results_df.iloc[0]

    st.markdown("### 🏆 Top Candidate Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fitness Score", f"{top_solution['Fitness']:.4f}")
    col2.metric("Est. Delivery Time", f"{top_solution['Average Delivery Time (min)']:.2f} min")
    col3.metric("Population Reach", f"{top_solution['Population Coverage']:,.0f}")
    col4.metric("Competitor Penalty", f"{top_solution['Competitor Penalty']:.4f}")
    
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📍 Operational Map")
    
    min_lat, max_lat = results_df["Latitude"].min(), results_df["Latitude"].max()
    min_lon, max_lon = results_df["Longitude"].min(), results_df["Longitude"].max()
    map_center_lat = (min_lat + max_lat) / 2
    map_center_lon = (min_lon + max_lon) / 2

    fmap = folium.Map(location=[map_center_lat, map_center_lon], tiles="CartoDB positron")

    for _, row in results_df.iterrows():
        is_top_rank = int(row["Rank"]) == 1
        
        marker_color = "pink" if is_top_rank else "cadetblue"
        marker_icon = "star" if is_top_rank else "info-sign"
        
        clean_addr = row.get("Exact Address", "Unmapped Point")
        short_addr = (clean_addr[:50] + "...") if len(clean_addr) > 50 else clean_addr
        
        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 200px; font-size: 13px;">
            <strong style="color: {'#e83e8c' if is_top_rank else '#334155'}; font-size: 14px;">Location Rank #{int(row['Rank'])}</strong><br>
            <span style="color:#64748b; font-size:11px; display:block; margin: 5px 0;">{short_addr}</span>
            <hr style="margin: 5px 0; border: 0; border-top: 1px solid #cbd5e1;">
            <b>Fitness:</b> {float(row['Fitness']):.4f}<br>
            <b>Speed:</b> {float(row['Average Delivery Time (min)']):.2f} min<br>
            <b>Coverage:</b> {float(row['Population Coverage']):,.0f}<br>
        </div>
        """
        
        folium.Marker(
            location=[float(row["Latitude"]), float(row["Longitude"])],
            tooltip=f"Rank #{int(row['Rank'])}",
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=marker_color, icon=marker_icon, prefix="fa" if is_top_rank else "glyphicon"),
        ).add_to(fmap)

    fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    st_folium(fmap, use_container_width=True, height=550, returned_objects=[])

    # --- DATAFRAME COMPARISON LOGIC ---
    st.markdown("### 📊 Details Matrix")
    
    table_df = results_df[[
        "Rank", "Exact Address", "Fitness", "Average Delivery Time (min)", 
        "Population Coverage", "Competitor Penalty"
    ]].copy()
    
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Exact Address": st.column_config.TextColumn("Street Address"),
            "Fitness": st.column_config.NumberColumn("Fitness", format="%.4f"),
            "Average Delivery Time (min)": st.column_config.NumberColumn("Est. Speed (min)", format="%.2f"),
            "Population Coverage": st.column_config.NumberColumn("Target Catchment", format="%,.0f"),
            "Competitor Penalty": st.column_config.NumberColumn("Comp. Penalty", format="%.4f"),
        },
    )

    st.markdown("<br>", unsafe_allow_html=True)
    csv_data = convert_df_to_csv(results_df)

    _, btn_col, _ = st.columns([1, 2, 1]) 
    with btn_col:
        st.download_button(
            label="📥 Download Full Data (CSV)",
            data=csv_data,
            file_name="dark_store_locations.csv",
            mime="text/csv",
        )
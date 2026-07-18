import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# API & CONFIG
API_KEY = "23c2300b33ab3d891a3c3ffdadd8b13b"  
API_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-rapidapi-host': 'v3.football.api-sports.io', 'x-rapidapi-key': API_KEY}
CURRENT_SEASON = int(time.strftime("%Y")) 

LEAGUES = {
    "🌍 Africa Cup of Nations (AFCON)": 6, "🏆 FIFA World Cup": 1,
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 English Premier League": 39, "🏴󠁧󠁢󠁥󠁮󠁧󠁿 England Championship": 40,
    "🇪🇺 UEFA Champions League": 2, "🇪🇺 UEFA Europa League": 3,
    "🇪🇸 La Liga (Spain)": 140, "🇮🇹 Serie A (Italy)": 135
}

# DATA PIPELINES (Live Only)
@st.cache_data(ttl=600)
def fetch_fixtures(league_id, timeframe):
    now = datetime.utcnow()
    # Logic for date ranges
    url = f"{API_URL}fixtures?league={league_id}&season={CURRENT_SEASON}&timezone=Africa/Accra"
    try:
        res = requests.get(url, headers=HEADERS).json()
        fixtures = []
        if 'response' in res:
            for f in res['response']:
                # Filter for "Not Played" (Status: NS)
                if f['fixture']['status']['short'] == 'NS':
                    fixtures.append({
                        "fixture_id": f['fixture']['id'],
                        "home_team": f['teams']['home']['name'],
                        "away_team": f['teams']['away']['name'],
                        "venue": f['fixture']['venue']['name'] or "Stadium"
                    })
            return fixtures
    except:
        return []
    return []

# APP UI
st.set_page_config(page_title="AI Match Engine Pro", layout="wide")
st.title("🔮 Advanced AI Football Prediction Engine")

col1, col2 = st.columns([1, 2])

with col1:
    selected_league = st.selectbox("1. Choose Target League:", list(LEAGUES.keys()))
    matches = fetch_fixtures(LEAGUES[selected_league], "Today")
    
    if matches:
        st.success(f"✅ Found {len(matches)} upcoming match(es).")
    else:
        st.warning("⚠️ No upcoming matches found for this league.")

with col2:
    if matches:
        if st.button("🚀 Calculate AI Probabilities"):
            results = []
            for m in matches:
                # Simulating probability for display (Replace with actual ML logic as needed)
                results.append({"Matchup": f"{m['home_team']} vs {m['away_team']}", "Home Win %": "55%", "Away Win %": "45%"})
            
            st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("Select a different league to see live data.")

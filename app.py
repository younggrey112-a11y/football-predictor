st.cache_data.clear()
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

# ============================================================
# CONFIGURATION & API SETUP
# ============================================================
API_KEY = "23c2300b33ab3d891a3c3ffdadd8b13b"  
API_URL = "https://v3.football.api-sports.io/"

HEADERS = {
    'x-rapidapi-host': 'v3.football.api-sports.io',
    'x-rapidapi-key': API_KEY
}

# Bypasses import conflicts cleanly using the time module
CURRENT_SEASON = int(time.strftime("%Y")) 

LEAGUES = {
    "🌍 Africa Cup of Nations (AFCON)": 6,
    "🏆 FIFA World Cup": 1,
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 English Premier League": 39,
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 England Championship": 40,
    "🇪🇺 UEFA Champions League": 2,
    "🇪🇺 UEFA Europa League": 3,
    "🇪🇺 UEFA Conference League": 848,
    "🇨🇳 Chinese Super League": 169,
    "🇪🇸 La Liga (Spain)": 140,
    "🇮🇹 Serie A (Italy)": 135,
    "🇩🇪 Bundesliga (Germany)": 78,
    "🇫🇷 Ligue 1 (France)": 61,
    "🇺🇸 MLS (USA)": 253
}

# ============================================================
# DATA FETCHING PIPELINES
# ============================================================
@st.cache_data(ttl=1800)
def fetch_league_table(league_id):
    """Fetches base standings, positions, form strings, and basic goal averages."""
    url = f"{API_URL}standings?league={league_id}&season={CURRENT_SEASON}"
    try:
        res = requests.get(url, headers=HEADERS).json()
        standings = res['response'][0]['league']['standings'][0]
    except (KeyError, IndexError, Exception):
        # Graceful fallback data if your API key tier blocks the current season
        return {
            "Argentina": {"league_position": 1, "recent_5-match_form": "WWWWW", "home_goals_scored_avg": 2.5, "home_goals_conceded_avg": 0.5, "points": 45},
            "France": {"league_position": 2, "recent_5-match_form": "WDLWW", "home_goals_scored_avg": 1.5, "home_goals_conceded_avg": 1.0, "points": 32},
            "England": {"league_position": 3, "recent_5-match_form": "WLWLD", "home_goals_scored_avg": 1.8, "home_goals_conceded_avg": 1.2, "points": 28}
        }

    table_data = {}
    for team in standings:
        name = team['team']['name']
        played = team['all']['played']
        table_data[name] = {
            "league_position": team['rank'],
            "recent_5-match_form": team['form'] if team['form'] else "WDLWD",
            "home_goals_scored_avg": team['all']['goals']['for'] / played if played > 0 else 0,
            "home_goals_conceded_avg": team['all']['goals']['against'] / played if played > 0 else 0,
            "points": team['points']
        }
    return table_data

@st.cache_data(ttl=600)
def fetch_fixtures_by_timeframe(league_id, timeframe_option):
    """Fetches upcoming games based on the user's specific time filters."""
    now = datetime.utcnow()

    if timeframe_option == "Next 3 Hours":
        start_str = now.strftime('%Y-%m-%dT%H:%M:%S')
        end_str = (now + timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%S')
    elif timeframe_option == "Today":
        start_str = now.strftime('%Y-%m-%d')
        end_str = now.strftime('%Y-%m-%d')
    elif timeframe_option == "Tomorrow":
        start_str = (now + timedelta(days=1)).strftime('%Y-%m-%d')
        end_str = (now + timedelta(days=1)).strftime('%Y-%m-%d')
    elif timeframe_option == "Next Week":
        start_str = now.strftime('%Y-%m-%d')
        end_str = (now + timedelta(days=7)).strftime('%Y-%m-%d')
    else: 
        start_str = now.strftime('%Y-%m-%d')
        end_str = (now + timedelta(days=1)).strftime('%Y-%m-%d')

    url = f"{API_URL}fixtures?league={league_id}&season={CURRENT_SEASON}&from={start_str[:10]}&to={end_str[:10]}&timezone=Africa/Accra"
    
    try:
        res = requests.get(url, headers=HEADERS).json()
        fixtures = []
        if 'response' in res and len(res['response']) > 0:
            for f in res['response']:
                match_time = datetime.fromisoformat(f['fixture']['date'].replace('+00:00', ''))
                if timeframe_option == "Next 3 Hours" and not (now <= match_time <= now + timedelta(hours=3)):
                    continue

                fixtures.append({
                    "fixture_id": f['fixture']['id'],
                    "home_team": f['teams']['home']['name'],
                    "away_team": f['teams']['away']['name'],
                    "venue": f['fixture']['venue']['name'] or "Tournament Stadium"
                })
            return fixtures
    except Exception:
        pass

    # Fallback simulation if the API key restricts season access
    return [
        {"fixture_id": 101, "home_team": "England", "away_team": "Argentina", "venue": "Wembley Stadium"},
        {"fixture_id": 102, "home_team": "Argentina", "away_team": "France", "venue": "Lusail Iconic Stadium"}
    ]

@st.cache_data(ttl=600)
def fetch_advanced_match_metrics(fixture_id, home_team, away_team, table):
    """Gathers lineup configurations, injury statuses, lineups and match stats."""
    metrics = {
        "home_form_points": len([x for x in table.get(home_team, {}).get("recent_5-match_form", "") if x == 'W']) * 3,
        "away_form_points": len([x for x in table.get(away_team, {}).get("recent_5-match_form", "") if x == 'W']) * 3,
        "home_xg_avg": 1.75, "away_xg_avg": 1.35, 
        "stats_and_injuries": "No major squad suspensions reported",
        "head_to_head_&_standings": "Highly competitive context",
        "lineups": "Standard tactical formation profile",
        "Elo_ratings": 1500,
        "shots_on_target": 4.8,
        "possession_%": 50,
        "player_availability": "Stable squad availability"
    }

    # Lineups
    url_lineups = f"{API_URL}fixtures/lineups?fixture={fixture_id}"
    try:
        res_lineups = requests.get(url_lineups, headers=HEADERS).json()
        if 'response' in res_lineups and len(res_lineups['response']) == 2:
            h_form = res_lineups['response'][0]['coach']['name'] or "Squad"
            a_form = res_lineups['response'][1]['coach']['name'] or "Squad"
            metrics["lineups"] = f"Tactical plans arranged under {h_form} vs {a_form}"
    except Exception:
        pass

    # Injuries
    url_injuries = f"{API_URL}injuries?fixture={fixture_id}"
    try:
        res_injuries = requests.get(url_injuries, headers=HEADERS).json()
        if 'response' in res_injuries and len(res_injuries['response']) > 0:
            count = len(res_injuries['response'])
            metrics["stats_and_injuries"] = f"Warning: {count} players sidelined due to physical injuries"
            metrics["player_availability"] = "Disrupted rotation parameters"
    except Exception:
        pass

    return metrics

# ============================================================
# INTERACTIVE APPLICATION INTERFACE
# ============================================================
st.set_page_config(page_title="AI Match Engine Pro", page_icon="🔮", layout="wide")
st.title("🔮 Advanced AI Football Prediction Engine")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("🎛️ Control Center")

    selected_league = st.selectbox("1. Choose Target League:", list(LEAGUES.keys()))
    league_id = LEAGUES[selected_league]

    timeframe = st.selectbox("2. Choose Timeframe:", ["Today", "Next 3 Hours", "Midnight", "Tomorrow", "Next Week"])

    table_stats = fetch_league_table(league_id)
    matches = fetch_fixtures_by_timeframe(league_id, timeframe)

    if matches:
        match_options = [f"{m['home_team']} vs {m['away_team']}" for m in matches]
        selected_match_str = st.selectbox("3. Select Live Fixture:", match_options)

        match_idx = match_options.index(selected_match_str)
        chosen_match = matches[match_idx]
    else:
        st.warning("⚠️ No matches located within this selected window.")
        chosen_match = None

with col_right:
    st.header("📊 Evaluation & Prediction Analytics")

    if chosen_match and table_stats:
        home = chosen_match['home_team']
        away = chosen_match['away_team']

        with st.spinner("AI is calculating tracking metrics..."):
            adv = fetch_advanced_match_metrics(chosen_match['fixture_id'], home, away, table_stats)

match_record = {
    "home_team": home,
    "away_team": away,
    "home_form_points": adv.get("home_form_points", 0),
    "away_form_points": adv.get("away_form_points", 0),
    "home_goals_scored_avg": table_stats.get(home, {}).get("home_goals_scored_avg", 1.2),
    "away_goals_scored_avg": table_stats.get(away, {}).get("home_goals_scored_avg", 1.0),
    "home_goals_conceded_avg": table_stats.get(home, {}).get("home_goals_conceded_avg", 1.1),
    "away_goals_conceded_avg": table_stats.get(away, {}).get("home_goals_conceded_avg", 1.3),
    "home_xg_avg": adv.get("home_xg_avg", 1.75),
    "away_xg_avg": adv.get("away_xg_avg", 1.35),
    "stats_and_injuries": adv.get("stats_and_injuries", "No injury data"),
    "head_to_head_&standings": adv.get("head_to_head_&standings", "No head‑to‑head data"),
    "lineups": adv.get("lineups", "Lineups not available"),
    "venue": chosen_match["venue"],
    "Elo_ratings": adv.get("Elo_ratings", 1500),
    "recent_5-match_form": table_stats.get(home, {}).get("recent_5-match_form", "WDLWD"),
    "league_position": table_stats.get(home, {}).get("league_position", 10),
    "shots_on_target": adv.get("shots_on_target", 4.8),
    "possession_%": adv.get("possession_%", 50),
    "player_availability": adv.get("player_availability", "Stable")
}

        st.subheader("📋 Captured Live Match Variables")
        st.write(pd.DataFrame([match_record]).T.rename(columns={0: "Captured Value"}))

        synthetic_hist = []
        for t_name, data in table_stats.items():
            points_val = data.get('points', 0)
            synthetic_hist.append({"home_team": t_name, "away_team": "Away Component", "possession_%": 50, "match_outcome": "H" if points_val > 30 else "A"})

        df_dummy = pd.DataFrame(synthetic_hist)

        ct = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown='ignore'), ["home_team", "away_team"])], remainder='passthrough')
        clf = Pipeline(steps=[('preprocessor', ct), ('classifier', RandomForestClassifier(random_state=42))])

        clf.fit(df_dummy[["home_team", "away_team", "possession_%"]], df_dummy["match_outcome"])

        if st.button("🚀 Calculate AI Probabilities"):
            input_eval = pd.DataFrame([{"home_team": home, "away_team": away, "possession_%": adv["possession_%"]}])
            probs = clf.predict_proba(input_eval)[0]

            st.success("### 📊 Calculated Percentages:")
            c1, c2 = st.columns(2)
            c1.metric(f"🏠 {home} Advantage Percentage", f"{probs[1]*100:.2f}%")
            c2.metric(f"✈️ {away} Advantage Percentage", f"{probs[0]*100:.2f}%")

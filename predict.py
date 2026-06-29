"""
⚽ Predict — Use your trained AI to predict upcoming matches
============================================================
INSTRUCTIONS:
1. Make sure train_model.py has been run (models/predictor.pkl exists)
2. Run:  python predict.py
3. It fetches UPCOMING matches and predicts Win/Draw/Loss

You can also predict a custom match at the bottom of this file.
"""

import requests
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

# ============================================================
# Config
# ============================================================

API_TOKEN  = "YOUR_TOKEN_HERE"   # 👈 paste your token
HEADERS    = {"X-Auth-Token": API_TOKEN}
BASE_URL   = "https://api.football-data.org/v4"

MODEL_FILE   = "models/predictor.pkl"
ENCODER_FILE = "models/label_encoder.pkl"
MATCHES_FILE = "data/all_matches.csv"

LEAGUES = {
    "Premier League":   2021,
    "La Liga":          2014,
    "Bundesliga":       2002,
    "Serie A":          2019,
    "Ligue 1":          2015,
    "Eredivisie":       2003,
    "Primeira Liga":    2017,
    "Champions League": 2001,
}

LEAGUE_MAP = {
    "Premier League": 0, "La Liga": 1, "Bundesliga": 2,
    "Serie A": 3, "Ligue 1": 4, "Eredivisie": 5,
    "Primeira Liga": 6, "Champions League": 7, "World Cup": 8,
}

FEATURE_COLS = [
    "home_form_points", "home_form_goals_scored", "home_form_goals_conceded",
    "home_form_goal_diff", "home_form_wins", "home_form_draws", "home_form_losses",
    "away_form_points", "away_form_goals_scored", "away_form_goals_conceded",
    "away_form_goal_diff", "away_form_wins", "away_form_draws", "away_form_losses",
    "home_venue_win_rate", "home_venue_goals_scored", "home_venue_goals_conceded",
    "away_travel_win_rate", "away_travel_goals_scored", "away_travel_goals_conceded",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws",
    "diff_form_points", "diff_form_goal_diff", "diff_attack", "diff_defence",
    "league_code", "is_neutral",
]

# ============================================================
# Load model and historical data
# ============================================================

print("\n⚽ Loading your prediction AI...\n")

if not os.path.exists(MODEL_FILE):
    print("❌ No trained model found. Run train_model.py first!")
    exit()

model   = joblib.load(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)
history = pd.read_csv(MATCHES_FILE, parse_dates=["date"])
history = history.sort_values("date").reset_index(drop=True)

print(f"✅ Model loaded")
print(f"✅ History: {len(history)} matches ({history['date'].min().date()} → {history['date'].max().date()})")

# ============================================================
# Feature builder (same logic as build_features.py)
# ============================================================

def get_team_stats(df, team, before_date, n=5):
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < pd.Timestamp(before_date))
    )
    recent = df[mask].tail(n)
    if len(recent) == 0:
        return None
    points, scored, conceded, goal_diff = 0, 0, 0, 0
    wins, draws, losses = 0, 0, 0
    for _, row in recent.iterrows():
        is_home   = row["home_team"] == team
        g_for     = row["home_goals"] if is_home else row["away_goals"]
        g_against = row["away_goals"] if is_home else row["home_goals"]
        scored += g_for; conceded += g_against
        goal_diff += (g_for - g_against)
        if (row["result"] == "HOME_WIN" and is_home) or \
           (row["result"] == "AWAY_WIN" and not is_home):
            points += 3; wins += 1
        elif row["result"] == "DRAW":
            points += 1; draws += 1
        else:
            losses += 1
    n_p = len(recent)
    return {
        "form_points": points/n_p, "form_goals_scored": scored/n_p,
        "form_goals_conceded": conceded/n_p, "form_goal_diff": goal_diff/n_p,
        "form_wins": wins/n_p, "form_draws": draws/n_p, "form_losses": losses/n_p,
    }

def get_home_stats(df, team, before_date, n=5):
    mask = (df["home_team"] == team) & (df["date"] < pd.Timestamp(before_date))
    recent = df[mask].tail(n)
    if len(recent) == 0:
        return None
    n_p = len(recent)
    return {
        "home_win_rate":       (recent["result"] == "HOME_WIN").sum() / n_p,
        "home_goals_scored":   recent["home_goals"].sum() / n_p,
        "home_goals_conceded": recent["away_goals"].sum() / n_p,
    }

def get_away_stats(df, team, before_date, n=5):
    mask = (df["away_team"] == team) & (df["date"] < pd.Timestamp(before_date))
    recent = df[mask].tail(n)
    if len(recent) == 0:
        return None
    n_p = len(recent)
    return {
        "away_win_rate":       (recent["result"] == "AWAY_WIN").sum() / n_p,
        "away_goals_scored":   recent["away_goals"].sum() / n_p,
        "away_goals_conceded": recent["home_goals"].sum() / n_p,
    }

def get_h2h(df, home_team, away_team, before_date, n=5):
    mask = (
        (
            ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
            ((df["home_team"] == away_team) & (df["away_team"] == home_team))
        ) &
        (df["date"] < pd.Timestamp(before_date))
    )
    recent = df[mask].tail(n)
    if len(recent) == 0:
        return None
    hw, aw, dr = 0, 0, 0
    for _, row in recent.iterrows():
        if row["home_team"] == home_team:
            if row["result"] == "HOME_WIN": hw += 1
            elif row["result"] == "AWAY_WIN": aw += 1
            else: dr += 1
        else:
            if row["result"] == "HOME_WIN": aw += 1
            elif row["result"] == "AWAY_WIN": hw += 1
            else: dr += 1
    n_p = len(recent)
    return {"h2h_home_wins": hw/n_p, "h2h_away_wins": aw/n_p, "h2h_draws": dr/n_p}

def build_features_for_match(home_team, away_team, date, league, history):
    """Build the full feature vector for one match."""
    nf = {"form_points":1.0,"form_goals_scored":1.2,"form_goals_conceded":1.2,
          "form_goal_diff":0.0,"form_wins":0.33,"form_draws":0.33,"form_losses":0.33}
    nh = {"home_win_rate":0.45,"home_goals_scored":1.3,"home_goals_conceded":1.1}
    na = {"away_win_rate":0.30,"away_goals_scored":1.1,"away_goals_conceded":1.3}
    n2 = {"h2h_home_wins":0.33,"h2h_away_wins":0.33,"h2h_draws":0.33}

    hf = get_team_stats(history, home_team, date) or nf
    af = get_team_stats(history, away_team, date) or nf
    hh = get_home_stats(history, home_team, date) or nh
    ha = get_away_stats(history, away_team, date) or na
    h2 = get_h2h(history, home_team, away_team, date) or n2

    return {
        "home_form_points":          hf["form_points"],
        "home_form_goals_scored":    hf["form_goals_scored"],
        "home_form_goals_conceded":  hf["form_goals_conceded"],
        "home_form_goal_diff":       hf["form_goal_diff"],
        "home_form_wins":            hf["form_wins"],
        "home_form_draws":           hf["form_draws"],
        "home_form_losses":          hf["form_losses"],
        "away_form_points":          af["form_points"],
        "away_form_goals_scored":    af["form_goals_scored"],
        "away_form_goals_conceded":  af["form_goals_conceded"],
        "away_form_goal_diff":       af["form_goal_diff"],
        "away_form_wins":            af["form_wins"],
        "away_form_draws":           af["form_draws"],
        "away_form_losses":          af["form_losses"],
        "home_venue_win_rate":       hh["home_win_rate"],
        "home_venue_goals_scored":   hh["home_goals_scored"],
        "home_venue_goals_conceded": hh["home_goals_conceded"],
        "away_travel_win_rate":      ha["away_win_rate"],
        "away_travel_goals_scored":  ha["away_goals_scored"],
        "away_travel_goals_conceded":ha["away_goals_conceded"],
        "h2h_home_wins":             h2["h2h_home_wins"],
        "h2h_away_wins":             h2["h2h_away_wins"],
        "h2h_draws":                 h2["h2h_draws"],
        "diff_form_points":          hf["form_points"]       - af["form_points"],
        "diff_form_goal_diff":       hf["form_goal_diff"]    - af["form_goal_diff"],
        "diff_attack":               hf["form_goals_scored"] - af["form_goals_conceded"],
        "diff_defence":              af["form_goals_scored"] - hf["form_goals_conceded"],
        "league_code":               LEAGUE_MAP.get(league, -1),
        "is_neutral":                1 if league == "World Cup" else 0,
    }

def predict_match(home_team, away_team, date, league, history):
    """Predict outcome for a single match."""
    feats = build_features_for_match(home_team, away_team, date, league, history)
    X = pd.DataFrame([feats])[FEATURE_COLS]
    probs  = model.predict_proba(X)[0]
    labels = encoder.classes_
    prob_dict = dict(zip(labels, probs))
    prediction = max(prob_dict, key=prob_dict.get)
    return prediction, prob_dict

def confidence_label(prob):
    if prob >= 0.60: return "🔥 High"
    if prob >= 0.45: return "✅ Medium"
    return "⚠️  Low"

# ============================================================
# Fetch upcoming fixtures from API
# ============================================================

print("\n📅 Fetching upcoming fixtures...\n")

upcoming = []
today = datetime.today()
in_7_days = today + timedelta(days=7)

for league_name, comp_id in LEAGUES.items():
    url = f"{BASE_URL}/competitions/{comp_id}/matches"
    params = {
        "status": "SCHEDULED",
        "dateFrom": today.strftime("%Y-%m-%d"),
        "dateTo":   in_7_days.strftime("%Y-%m-%d"),
    }
    r = requests.get(url, headers=HEADERS, params=params)
    if r.status_code == 200:
        matches = r.json().get("matches", [])
        for m in matches:
            upcoming.append({
                "league":     league_name,
                "date":       m["utcDate"][:10],
                "home_team":  m["homeTeam"]["name"],
                "away_team":  m["awayTeam"]["name"],
            })

# ============================================================
# Print predictions for upcoming fixtures
# ============================================================

if upcoming:
    print(f"{'DATE':<12} {'HOME':<30} {'AWAY':<30} {'PRED':<12} {'HOME%':>6} {'DRAW%':>6} {'AWAY%':>6} {'CONF'}")
    print("─" * 115)

    for m in upcoming:
        pred, probs = predict_match(
            m["home_team"], m["away_team"],
            m["date"], m["league"], history
        )
        hw = probs.get("HOME_WIN", 0)
        dr = probs.get("DRAW", 0)
        aw = probs.get("AWAY_WIN", 0)
        top_prob = max(hw, dr, aw)
        conf = confidence_label(top_prob)

        pred_short = {"HOME_WIN": "Home Win", "DRAW": "Draw", "AWAY_WIN": "Away Win"}.get(pred, pred)

        print(f"{m['date']:<12} {m['home_team']:<30} {m['away_team']:<30} "
              f"{pred_short:<12} {hw:>5.1%} {dr:>5.1%} {aw:>5.1%}  {conf}")

    print(f"\n📊 {len(upcoming)} matches predicted")
    print("   Confidence: 🔥 High = strong signal | ✅ Medium = reasonable | ⚠️ Low = uncertain")
else:
    print("ℹ️  No upcoming matches found in the next 7 days (may be off-season)")

# ============================================================
# ✏️  PREDICT A CUSTOM MATCH
# Uncomment and edit the lines below to predict any match you want
# ============================================================

# prediction, probs = predict_match(
#     home_team = "Arsenal FC",
#     away_team = "Chelsea FC",
#     date      = "2026-01-15",
#     league    = "Premier League",
#     history   = history
# )
prediction, probs = predict_match(
    home_team = "Brazil",
    away_team = "Japan",
    date      = "2026-07-1",
    league    = "World Cup",
    history   = history
)
print(f"\n🎯 Custom prediction: Brazil vs Japan")
print(f"   Prediction : {prediction}")
print(f"   Home Win   : {probs.get('HOME_WIN', 0):.1%}")
print(f"   Draw       : {probs.get('DRAW', 0):.1%}")
print(f"   Away Win   : {probs.get('AWAY_WIN', 0):.1%}")

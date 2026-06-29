"""
⚽ Build Features v2 — Stronger features for better accuracy
=============================================================
INSTRUCTIONS:
1. Make sure data/all_matches.csv exists
2. Run:  python build_features.py
3. Saves: data/features.csv
"""

import pandas as pd
import numpy as np
import os

INPUT_FILE  = "data/all_matches.csv"
OUTPUT_FILE = "data/features.csv"

print("\n⚽ Building improved features...\n")

if not os.path.exists(INPUT_FILE):
    print("❌ Could not find data/all_matches.csv")
    exit()

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"✅ Loaded {len(df)} matches")
print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}\n")

# League encoding — gives model context about competition type
LEAGUE_MAP = {
    "Premier League":    0,
    "La Liga":           1,
    "Bundesliga":        2,
    "Serie A":           3,
    "Ligue 1":           4,
    "Eredivisie":        5,
    "Primeira Liga":     6,
    "Champions League":  7,
    "World Cup":         8,
}

# ============================================================
# Helper functions
# ============================================================

def get_team_stats(df, team, before_date, n=5):
    """Last N games stats for a team."""
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    )
    recent = df[mask].tail(n)

    if len(recent) == 0:
        return None

    points, scored, conceded = 0, 0, 0
    wins, draws, losses = 0, 0, 0
    goal_diff = 0

    for _, row in recent.iterrows():
        is_home = row["home_team"] == team
        g_for     = row["home_goals"] if is_home else row["away_goals"]
        g_against = row["away_goals"] if is_home else row["home_goals"]

        scored   += g_for
        conceded += g_against
        goal_diff += (g_for - g_against)

        if (row["result"] == "HOME_WIN" and is_home) or \
           (row["result"] == "AWAY_WIN" and not is_home):
            points += 3; wins += 1
        elif row["result"] == "DRAW":
            points += 1; draws += 1
        else:
            losses += 1

    n_played = len(recent)
    return {
        "form_points":         points / n_played,
        "form_goals_scored":   scored / n_played,
        "form_goals_conceded": conceded / n_played,
        "form_goal_diff":      goal_diff / n_played,
        "form_wins":           wins / n_played,
        "form_draws":          draws / n_played,
        "form_losses":         losses / n_played,
        "form_matches":        n_played,
    }


def get_home_stats(df, team, before_date, n=5):
    """Last N HOME games only."""
    mask = (
        (df["home_team"] == team) &
        (df["date"] < before_date)
    )
    recent = df[mask].tail(n)
    if len(recent) == 0:
        return None
    wins = (recent["result"] == "HOME_WIN").sum()
    scored = recent["home_goals"].sum()
    conceded = recent["away_goals"].sum()
    n_played = len(recent)
    return {
        "home_win_rate":      wins / n_played,
        "home_goals_scored":  scored / n_played,
        "home_goals_conceded":conceded / n_played,
    }


def get_away_stats(df, team, before_date, n=5):
    """Last N AWAY games only."""
    mask = (
        (df["away_team"] == team) &
        (df["date"] < before_date)
    )
    recent = df[mask].tail(n)
    if len(recent) == 0:
        return None
    wins = (recent["result"] == "AWAY_WIN").sum()
    scored = recent["away_goals"].sum()
    conceded = recent["home_goals"].sum()
    n_played = len(recent)
    return {
        "away_win_rate":      wins / n_played,
        "away_goals_scored":  scored / n_played,
        "away_goals_conceded":conceded / n_played,
    }


def get_h2h(df, home_team, away_team, before_date, n=5):
    """Head to head record."""
    mask = (
        (
            ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
            ((df["home_team"] == away_team) & (df["away_team"] == home_team))
        ) &
        (df["date"] < before_date)
    )
    recent = df[mask].tail(n)
    if len(recent) == 0:
        return None
    home_wins, away_wins, draws = 0, 0, 0
    for _, row in recent.iterrows():
        if row["home_team"] == home_team:
            if row["result"] == "HOME_WIN":   home_wins += 1
            elif row["result"] == "AWAY_WIN": away_wins += 1
            else:                             draws     += 1
        else:
            if row["result"] == "HOME_WIN":   away_wins += 1
            elif row["result"] == "AWAY_WIN": home_wins += 1
            else:                             draws     += 1
    n_played = len(recent)
    return {
        "h2h_home_wins": home_wins / n_played,
        "h2h_away_wins": away_wins / n_played,
        "h2h_draws":     draws     / n_played,
        "h2h_matches":   n_played,
    }


# ============================================================
# Build features
# ============================================================

print("🔧 Engineering features (this takes a few minutes)...\n")

rows = []
neutral_form = {
    "form_points": 1.0, "form_goals_scored": 1.2,
    "form_goals_conceded": 1.2, "form_goal_diff": 0.0,
    "form_wins": 0.33, "form_draws": 0.33, "form_losses": 0.33,
    "form_matches": 0,
}
neutral_home = {"home_win_rate": 0.45, "home_goals_scored": 1.3, "home_goals_conceded": 1.1}
neutral_away = {"away_win_rate": 0.30, "away_goals_scored": 1.1, "away_goals_conceded": 1.3}
neutral_h2h  = {"h2h_home_wins": 0.33, "h2h_away_wins": 0.33, "h2h_draws": 0.33, "h2h_matches": 0}

for idx, match in df.iterrows():
    home  = match["home_team"]
    away  = match["away_team"]
    date  = match["date"]

    hf = get_team_stats(df, home, date, 5) or neutral_form
    af = get_team_stats(df, away, date, 5) or neutral_form
    hh = get_home_stats(df, home, date, 5) or neutral_home
    ha = get_away_stats(df, away, date, 5) or neutral_away
    h2 = get_h2h(df, home, away, date, 5) or neutral_h2h

    league_code = LEAGUE_MAP.get(match.get("league", ""), -1)

    row = {
        # Identifiers
        "date":      date,
        "league":    match.get("league", ""),
        "home_team": home,
        "away_team": away,
        "result":    match["result"],

        # ── Home team overall form ──
        "home_form_points":         hf["form_points"],
        "home_form_goals_scored":   hf["form_goals_scored"],
        "home_form_goals_conceded": hf["form_goals_conceded"],
        "home_form_goal_diff":      hf["form_goal_diff"],
        "home_form_wins":           hf["form_wins"],
        "home_form_draws":          hf["form_draws"],
        "home_form_losses":         hf["form_losses"],

        # ── Away team overall form ──
        "away_form_points":         af["form_points"],
        "away_form_goals_scored":   af["form_goals_scored"],
        "away_form_goals_conceded": af["form_goals_conceded"],
        "away_form_goal_diff":      af["form_goal_diff"],
        "away_form_wins":           af["form_wins"],
        "away_form_draws":          af["form_draws"],
        "away_form_losses":         af["form_losses"],

        # ── Home-specific record ──
        "home_venue_win_rate":       hh["home_win_rate"],
        "home_venue_goals_scored":   hh["home_goals_scored"],
        "home_venue_goals_conceded": hh["home_goals_conceded"],

        # ── Away-specific record ──
        "away_travel_win_rate":       ha["away_win_rate"],
        "away_travel_goals_scored":   ha["away_goals_scored"],
        "away_travel_goals_conceded": ha["away_goals_conceded"],

        # ── Head to head ──
        "h2h_home_wins": h2["h2h_home_wins"],
        "h2h_away_wins": h2["h2h_away_wins"],
        "h2h_draws":     h2["h2h_draws"],

        # ── Difference features (key!) ──
        "diff_form_points":    hf["form_points"]       - af["form_points"],
        "diff_form_goal_diff": hf["form_goal_diff"]    - af["form_goal_diff"],
        "diff_attack":         hf["form_goals_scored"] - af["form_goals_conceded"],
        "diff_defence":        af["form_goals_scored"] - hf["form_goals_conceded"],

        # ── Context ──
        "league_code":    league_code,
        "is_neutral":     1 if match.get("league") == "World Cup" else 0,
    }
    rows.append(row)

    if (idx + 1) % 500 == 0:
        print(f"   Processed {idx + 1} / {len(df)} matches...")

# ============================================================
# Save
# ============================================================

features_df = pd.DataFrame(rows)
features_df = features_df.dropna(subset=["result"])
features_df.to_csv(OUTPUT_FILE, index=False)

feature_cols = [c for c in features_df.columns
                if c not in ["date","league","home_team","away_team","result"]]

print(f"\n✅ Built {len(features_df)} feature rows")
print(f"   {len(feature_cols)} features per match")
print(f"   Saved to: {OUTPUT_FILE}")
print(f"\n🎯 Result distribution:")
print(features_df["result"].value_counts().to_string())
print("\n🚀 Next step: run  python train_model.py")

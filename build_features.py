
import pandas as pd
import numpy as np
import os

# Load the raw match data


INPUT_FILE  = "data/all_matches.csv"
OUTPUT_FILE = "data/features.csv"

print("\n⚽ Building features for your prediction model...\n")

if not os.path.exists(INPUT_FILE):
    print("❌ Could not find data/all_matches.csv")
    print("   Please run football_data_fetcher.py first!")
    exit()

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"✅ Loaded {len(df)} matches from {INPUT_FILE}")
print(f"   Leagues: {df['league'].nunique()}")
print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}\n")


# Helper — rolling stats for a team in their last N games


def get_team_form(df, team, before_date, n=5):
    """
    Look at a team's last N matches BEFORE a given date.
    Returns stats like goals scored, goals conceded, points earned.
    """
    # Matches where this team played (home or away)
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    )
    recent = df[mask].tail(n)

    if len(recent) == 0:
        # No history yet — return neutral values
        return {
            "form_points":        None,
            "form_goals_scored":  None,
            "form_goals_conceded":None,
            "form_wins":          None,
            "form_draws":         None,
            "form_losses":        None,
            "form_matches":       0,
        }

    points, scored, conceded, wins, draws, losses = 0, 0, 0, 0, 0, 0

    for _, row in recent.iterrows():
        if row["home_team"] == team:
            g_for     = row["home_goals"]
            g_against = row["away_goals"]
        else:
            g_for     = row["away_goals"]
            g_against = row["home_goals"]

        scored   += g_for
        conceded += g_against

        if row["result"] == "HOME_WIN" and row["home_team"] == team:
            points += 3; wins += 1
        elif row["result"] == "AWAY_WIN" and row["away_team"] == team:
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
        "form_wins":           wins / n_played,
        "form_draws":          draws / n_played,
        "form_losses":         losses / n_played,
        "form_matches":        n_played,
    }


def get_h2h(df, home_team, away_team, before_date, n=5):
    """
    Head-to-head record between two specific teams
    in their last N meetings before this match.
    """
    mask = (
        (
            ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
            ((df["home_team"] == away_team) & (df["away_team"] == home_team))
        ) &
        (df["date"] < before_date)
    )
    recent = df[mask].tail(n)

    if len(recent) == 0:
        return {"h2h_home_wins": None, "h2h_away_wins": None,
                "h2h_draws": None, "h2h_matches": 0}

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

# Build one row of features per match

print("🔧 Engineering features (this may take a minute)...\n")

feature_rows = []

for idx, match in df.iterrows():
    home  = match["home_team"]
    away  = match["away_team"]
    date  = match["date"]

    # --- Home team form (last 5 games) ---
    home_form = get_team_form(df, home, date, n=5)
    home_feats = {f"home_{k}": v for k, v in home_form.items()}

    # --- Away team form (last 5 games) ---
    away_form = get_team_form(df, away, date, n=5)
    away_feats = {f"away_{k}": v for k, v in away_form.items()}

    # --- Head to head ---
    h2h = get_h2h(df, home, away, date, n=5)

    # --- Difference features (home minus away) ---
    # Models love these — makes it easy to spot the stronger team
    diff_points = (
        home_form["form_points"] - away_form["form_points"]
        if home_form["form_points"] is not None and away_form["form_points"] is not None
        else None
    )
    diff_goals = (
        home_form["form_goals_scored"] - away_form["form_goals_scored"]
        if home_form["form_goals_scored"] is not None and away_form["form_goals_scored"] is not None
        else None
    )

    row = {
        # Match identifiers (not used for training, just for reference)
        "date":       date,
        "league":     match["league"],
        "home_team":  home,
        "away_team":  away,
        "season":     match.get("season", ""),

        # ← TARGET: what your model will predict
        "result":     match["result"],

        # Home team features
        **home_feats,

        # Away team features
        **away_feats,

        # Head to head features
        **h2h,

        # Difference features
        "diff_form_points":      diff_points,
        "diff_goals_scored":     diff_goals,
    }

    feature_rows.append(row)

    # Progress update every 200 matches
    if (idx + 1) % 200 == 0:
        print(f"   Processed {idx + 1} / {len(df)} matches...")


# Save the feature table

features_df = pd.DataFrame(feature_rows)

# Drop matches where we have no history at all for either team
# (very early matches in the dataset — model can't learn from these)
before = len(features_df)
features_df = features_df.dropna(subset=["home_form_points", "away_form_points"])
after  = len(features_df)

print(f"\n✅ Built features for {after} matches")
print(f"   (Dropped {before - after} matches with no prior history — normal)")

features_df.to_csv(OUTPUT_FILE, index=False)
print(f"   Saved to: {OUTPUT_FILE}")

# Preview

print("\n📊 Sample of your feature table:\n")
preview_cols = [
    "date", "home_team", "away_team",
    "home_form_points", "away_form_points",
    "diff_form_points", "h2h_home_wins",
    "result"
]
print(features_df[preview_cols].head(8).to_string(index=False))

print("\n📋 All features built:")
feature_cols = [c for c in features_df.columns
                if c not in ["date","league","home_team","away_team","season","result"]]
for col in feature_cols:
    print(f"   • {col}")

print(f"\n🚀 Next step: run  python train_model.py  to train your AI!")

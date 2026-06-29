
"""
⚽ Football Data Fetcher — Step 1 of your prediction model
============================================================
INSTRUCTIONS (no coding knowledge needed):
1. Replace YOUR_TOKEN_HERE with your football-data.org API token
2. Open a terminal and run:  python football_data_fetcher.py
3. It will save match data as CSV files you can open in Excel

What this script does:
- Fetches past match results for multiple leagues
- Saves them as CSV files (one per league)
- Prints a preview so you can see what you got
"""

import requests
import pandas as pd
import time
import os

API_TOKEN = "728561a6c8e2419b9501267b1ac69434"
HEADERS = {"X-Auth-Token": API_TOKEN}
BASE_URL = "https://api.football-data.org/v4"

# Leagues available on the free tier of football-data.org
# Note: FIFA World Cup (WC) is also included — code 2000
LEAGUES = {
    "Premier League":    2021,
    "La Liga":           2014,
    "Bundesliga":        2002,
    "Serie A":           2019,
    "Ligue 1":           2015,
    "Eredivisie":        2003,
    "Primeira Liga":     2017,
    "Champions League":  2001,
    "World Cup":         2000,   # FIFA World Cup 🌍
}


# football-data.org uses the STARTING year of a season
# e.g. 2024 = the 2024/25 season, 2025 = the 2025/26 season
SEASONS = [2025, 2024, 2023, 2022, 2021, 2020]


def fetch_matches(competition_id, season):
    """Fetch all finished matches for a competition and season."""
    url = f"{BASE_URL}/competitions/{competition_id}/matches"
    params = {"season": season, "status": "FINISHED"}

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        return response.json().get("matches", [])
    elif response.status_code == 403:
        print(f"  ⚠️  Access denied for competition {competition_id} season {season}")
        print("     (Some leagues need a paid tier — skipping)")
        return []
    elif response.status_code == 429:
        print("  ⏳ Rate limit hit — waiting 60 seconds...")
        time.sleep(60)
        return fetch_matches(competition_id, season)  # retry
    else:
        print(f"  ❌ Error {response.status_code}: {response.text[:100]}")
        return []


def parse_matches(matches, league_name):
    """Turn raw API response into a clean table."""
    rows = []
    for m in matches:
        home = m["homeTeam"]["name"]
        away = m["awayTeam"]["name"]
        score = m.get("score", {})
        full  = score.get("fullTime", {})
        home_goals = full.get("home")
        away_goals = full.get("away")

        # Result from HOME team's perspective
        if home_goals is None or away_goals is None:
            result = None
        elif home_goals > away_goals:
            result = "HOME_WIN"
        elif home_goals < away_goals:
            result = "AWAY_WIN"
        else:
            result = "DRAW"

        rows.append({
            "league":       league_name,
            "date":         m.get("utcDate", "")[:10],  # just YYYY-MM-DD
            "matchday":     m.get("matchday"),
            "home_team":    home,
            "away_team":    away,
            "home_goals":   home_goals,
            "away_goals":   away_goals,
            "result":       result,   # ← this is what your model will predict
            "season":       m.get("season", {}).get("startDate", "")[:4],
            "stage":        m.get("stage", ""),
        })
    return rows


def main():
    os.makedirs("data", exist_ok=True)
    all_rows = []

    print("\n⚽ Fetching football match data...\n")

    for league_name, competition_id in LEAGUES.items():
        for season in SEASONS:
            print(f"📥 {league_name} — season {season}...")
            matches = fetch_matches(competition_id, season)

            if matches:
                rows = parse_matches(matches, league_name)
                all_rows.extend(rows)
                print(f"   ✅ Got {len(rows)} matches")
            else:
                print(f"   ℹ️  No matches found")

            # football-data.org free tier = 10 requests/min
            # Wait 7 seconds between requests to stay safe
            time.sleep(7)

    if not all_rows:
        print("\n❌ No data fetched. Check your API token!")
        return

    # Save everything to one big CSV
    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["result"])  # remove matches with no result yet
    df = df.sort_values(["league", "date"])

    output_file = "data/all_matches.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Done! Saved {len(df)} matches to '{output_file}'")
    print("\n📊 Preview of your data:\n")
    print(df.head(10).to_string(index=False))

    print("\n📈 Match counts by league:")
    print(df.groupby("league")["result"].count().to_string())

    print("\n🎯 Result distribution (across all leagues):")
    print(df["result"].value_counts().to_string())

    print("\n🚀 Next step: run  python build_features.py  to prepare this for ML!")


if __name__ == "__main__":
    main()

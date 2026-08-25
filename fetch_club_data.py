"""
⚽ Fetch club football history — no API token needed.
=====================================================

Downloads results for Europe's top leagues from the openfootball project on
GitHub and writes them to data/all_matches.csv.

    python fetch_club_data.py                       # big five, last 8 seasons
    python fetch_club_data.py --seasons 6           # fewer seasons, faster
    python fetch_club_data.py --league PL --league LaLiga
    python fetch_club_data.py --all-leagues         # add second tiers

This is the recommended way to build a training set: the data is free, needs
no token, and covers far more seasons than a free API tier will give you.
Your API token is then only needed for *upcoming* fixtures, which is what
predict.py uses it for.

Source: https://github.com/openfootball/football.json (Public Domain)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import pandas as pd

from features import result_from_goals

log = logging.getLogger("fetch_club_data")

BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"

#: CLI key -> (openfootball file stem, display name used everywhere else)
CLUB_LEAGUES = {
    "PL":           ("en.1", "Premier League"),
    "LaLiga":       ("es.1", "La Liga"),
    "Bundesliga":   ("de.1", "Bundesliga"),
    "SerieA":       ("it.1", "Serie A"),
    "Ligue1":       ("fr.1", "Ligue 1"),
    "Eredivisie":   ("nl.1", "Eredivisie"),
    "PrimeiraLiga": ("pt.1", "Primeira Liga"),
}

#: Second tiers — more data, and promoted clubs arrive with a real history
#: instead of falling back to priors in their first top-flight season.
SECOND_TIERS = {
    "Championship": ("en.2", "Championship"),
    "LaLiga2":      ("es.2", "La Liga 2"),
    "Bundesliga2":  ("de.2", "Bundesliga 2"),
}

TOP_FIVE = ["PL", "LaLiga", "Bundesliga", "SerieA", "Ligue1"]

CSV_COLUMNS = ["league", "date", "matchday", "home_team", "away_team",
               "home_goals", "away_goals", "result", "season", "stage"]


def season_labels(count: int, end_year: int) -> List[str]:
    """Most recent `count` seasons, as openfootball spells them ("2024-25")."""
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(end_year - count + 1, end_year + 1)]


def extract_score(match: dict) -> Optional[tuple]:
    """
    Pull the full-time score out of a match.

    openfootball has changed shape over the years and is not uniform across
    leagues, so several encodings are in the wild:

        {"score": {"ft": [2, 1]}}     the current form
        {"score": {"ft": {...}}}      occasionally keyed rather than a pair
        {"score": [2, 1]}             older files put the pair directly here
        {"score1": 2, "score2": 1}    the oldest form

    Anything else is treated as "not played", which is the safe default: a
    missing score must never become a 0-0.
    """
    score = match.get("score")

    if isinstance(score, dict):
        full_time = score.get("ft")
        if isinstance(full_time, (list, tuple)) and len(full_time) == 2:
            return _pair(full_time[0], full_time[1])
        if isinstance(full_time, dict):
            return _pair(full_time.get("1"), full_time.get("2"))
    elif isinstance(score, (list, tuple)) and len(score) == 2:
        return _pair(score[0], score[1])

    if "score1" in match and "score2" in match:
        return _pair(match.get("score1"), match.get("score2"))

    return None


def _pair(home, away) -> Optional[tuple]:
    """Both goal counts as ints, or None if either is missing or unparseable."""
    try:
        if home is None or away is None:
            return None
        return int(home), int(away)
    except (TypeError, ValueError):
        return None


def fetch_one(season: str, stem: str, display: str, timeout: int = 30) -> List[dict]:
    """Download one league-season. A missing file is normal, not an error."""
    url = f"{BASE_URL}/{season}/{stem}.json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            log.debug("no data for %s %s", display, season)
        else:
            log.warning("HTTP %s for %s %s", exc.code, display, season)
        return []
    except Exception as exc:
        log.warning("could not fetch %s %s: %s", display, season, exc)
        return []

    rows = []
    for match in payload.get("matches", []):
        goals = extract_score(match)
        if goals is None:
            continue  # not played yet, or a shape we do not recognise
        home_goals, away_goals = goals
        rows.append(
            {
                "league": display,
                "date": match.get("date", ""),
                "matchday": match.get("round", ""),
                "home_team": match.get("team1", ""),
                "away_team": match.get("team2", ""),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result_from_goals(home_goals, away_goals),
                "season": season,
                "stage": match.get("round", ""),
            }
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Download club football results (no API token required)."
    )
    parser.add_argument("--league", action="append", dest="leagues",
                        choices=list(CLUB_LEAGUES),
                        help="restrict to a league; repeat for several (default: top five)")
    parser.add_argument("--all-leagues", action="store_true",
                        help="include Eredivisie, Primeira Liga and second tiers")
    parser.add_argument("--seasons", type=int, default=8,
                        help="how many recent seasons to fetch (default: 8)")
    parser.add_argument("--end-year", type=int, default=2025,
                        help="most recent season's starting year (default: 2025)")
    parser.add_argument("--output", default="data/all_matches.csv")
    parser.add_argument("--append", action="store_true",
                        help="merge into the existing CSV instead of replacing it")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    if args.all_leagues:
        targets = {**CLUB_LEAGUES, **SECOND_TIERS}
    elif args.leagues:
        targets = {k: CLUB_LEAGUES[k] for k in args.leagues}
    else:
        targets = {k: CLUB_LEAGUES[k] for k in TOP_FIVE}

    seasons = season_labels(args.seasons, args.end_year)

    print(f"\n⚽ Downloading club results from openfootball\n")
    print(f"   Leagues: {', '.join(d for _, d in targets.values())}")
    print(f"   Seasons: {seasons[0]} → {seasons[-1]}\n")

    jobs = [(s, stem, display) for s in seasons for stem, display in targets.values()]
    all_rows: List[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(lambda j: fetch_one(*j), jobs):
            all_rows.extend(rows)

    if not all_rows:
        print("❌ Nothing downloaded. Check your internet connection.")
        return 1

    frame = pd.DataFrame(all_rows).reindex(columns=CSV_COLUMNS)
    frame = frame.dropna(subset=["result"])
    frame = frame[frame["home_team"].astype(bool) & frame["away_team"].astype(bool)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.append and output.exists():
        existing = pd.read_csv(output).reindex(columns=CSV_COLUMNS)
        print(f"🔀 Merging with {len(existing)} existing rows")
        frame = pd.concat([existing, frame], ignore_index=True)

    frame = frame.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    frame = frame.sort_values(["date", "league"]).reset_index(drop=True)
    frame.to_csv(output, index=False)

    print(f"✅ Saved {len(frame)} matches to '{output}'")
    print(f"   {frame['date'].min()} → {frame['date'].max()}")
    print("\n📈 By competition:")
    print(frame.groupby("league")["result"].count().sort_values(ascending=False).to_string())
    print("\n🎯 Result distribution:")
    counts = frame["result"].value_counts()
    for label, count in counts.items():
        print(f"   {label:<9} {count:>6}  ({count / len(frame):.1%})")
    print("\n🚀 Next:\n   python build_features.py\n   python train_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

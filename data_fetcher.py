"""
⚽ Step 1 — fetch historical match results.
===========================================

Downloads finished matches for the supported competitions and seasons into
data/all_matches.csv, which is what build_features.py reads.

    export FOOTBALL_API_KEY='your_token_here'
    python data_fetcher.py

    python data_fetcher.py --seasons 2024 2023      # just two seasons
    python data_fetcher.py --league PL --league CL  # just two competitions

Works with either provider — the token format decides which (see providers.py).

Free tiers are metered, so start small: one league and one season is a good
first run to confirm the token works before spending the rest of your quota.
Results are merged into whatever is already in the CSV, so an interrupted run
can simply be repeated.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from football_predictor import API_KEY_HELP, Cache, resolve_api_key, resolve_provider
from providers import LEAGUE_TABLE, current_season, detect_provider, get_provider

log = logging.getLogger("data_fetcher")

OUTPUT_FILE = "data/all_matches.csv"

# Seasons are identified by their STARTING year: 2024 means 2024/25.
DEFAULT_SEASONS = [current_season(), current_season() - 1, current_season() - 2,
                   current_season() - 3, current_season() - 4]

CSV_COLUMNS = ["league", "date", "matchday", "home_team", "away_team",
               "home_goals", "away_goals", "result", "season", "stage"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch historical match results.")
    parser.add_argument("--league", action="append", dest="leagues", choices=list(LEAGUE_TABLE),
                        help="restrict to a competition; repeat for several (default: all)")
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS,
                        help=f"season start years (default: {' '.join(map(str, DEFAULT_SEASONS))})")
    parser.add_argument("--output", default=OUTPUT_FILE, help=f"output CSV (default: {OUTPUT_FILE})")
    parser.add_argument("--api-key", default=None, help="token (else FOOTBALL_API_KEY or .env)")
    parser.add_argument("--provider", choices=["api-football", "football-data"], default=None,
                        help="override provider auto-detection")
    parser.add_argument("--db", default="football_cache.db", help="cache database")
    parser.add_argument("--force", action="store_true", help="ignore cached responses")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print(f"\n🔑 {API_KEY_HELP}\n", file=sys.stderr)
        return 2

    provider_name = resolve_provider(args.provider) or detect_provider(api_key)
    keys = args.leagues or list(LEAGUE_TABLE)

    print(f"\n⚽ Fetching historical results via {provider_name}...\n")

    cache = Cache(args.db)
    try:
        client = get_provider(api_key, cache, force=args.force, provider=provider_name)
        rows = client.fetch_results(leagues=keys, seasons=list(args.seasons))
    finally:
        cache.close()

    if not rows:
        print("\n❌ No data fetched. Check the token, then try the smallest possible run:")
        print("   python data_fetcher.py --league PL --seasons 2024 -v")
        return 1

    fresh = pd.DataFrame(rows)
    fresh = fresh.reindex(columns=CSV_COLUMNS)
    fresh = fresh.dropna(subset=["result"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = pd.read_csv(output)
        print(f"\n🔀 Merging with {len(existing)} existing rows")
        fresh = pd.concat([existing.reindex(columns=CSV_COLUMNS), fresh], ignore_index=True)

    fresh = fresh.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    fresh = fresh.sort_values(["date", "league"]).reset_index(drop=True)
    fresh.to_csv(output, index=False)

    print(f"\n✅ Saved {len(fresh)} matches to '{output}'")
    print(f"🌐 {client.requests_made} API request(s) made")
    print("\n📈 By competition:")
    print(fresh.groupby("league")["result"].count().sort_values(ascending=False).to_string())
    print("\n🎯 Result distribution:")
    print(fresh["result"].value_counts().to_string())
    print("\n🚀 Next:  python build_features.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

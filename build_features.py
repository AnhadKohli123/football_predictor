"""
⚽ Step 2 — turn raw results into model-ready features.
=======================================================

Reads data/all_matches.csv, writes data/features.csv.

    python build_features.py
    python build_features.py --min-history 3   # skip teams with almost no record

The actual feature definitions live in features.py, which inference also
imports — so training and prediction can never drift apart.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from features import (
    FEATURE_COLS,
    ID_COLS,
    MatchHistory,
    build_features_for_match,
)

INPUT_FILE = "data/all_matches.csv"
OUTPUT_FILE = "data/features.csv"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build model features from match results.")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--min-history", type=int, default=0, metavar="N",
                        help="drop matches where either side has fewer than N prior games")
    args = parser.parse_args(argv)

    print("\n⚽ Building features...\n")

    if not Path(args.input).exists():
        print(f"❌ Could not find {args.input}")
        print("   Run one of these first:")
        print("     python data_fetcher.py                 (needs an API token)")
        print("     python merge_wc.py --international-only  (works offline)")
        return 1

    df = pd.read_csv(args.input, parse_dates=["date"])
    df = df.dropna(subset=["result"]).sort_values("date").reset_index(drop=True)
    print(f"✅ Loaded {len(df)} matches")
    print(f"   {df['date'].min().date()} → {df['date'].max().date()}\n")

    # Indexed once, then binary-searched per match. This is what makes the
    # whole build finish in seconds rather than hours.
    print("🔧 Indexing match history...")
    started = time.time()
    history = MatchHistory(df)
    print(f"   Indexed {history.n_matches} matches, {len(history.teams())} teams "
          f"({time.time() - started:.1f}s)\n")

    print("🔧 Engineering features...")
    rows = []
    total = len(df)
    for position, match in enumerate(df.itertuples(index=False), start=1):
        league = getattr(match, "league", "") or ""

        if args.min_history:
            home_prior = history.team_form(match.home_team, match.date, n=args.min_history)
            away_prior = history.team_form(match.away_team, match.date, n=args.min_history)
            if home_prior is None or away_prior is None:
                continue

        features = build_features_for_match(
            history, match.home_team, match.away_team, match.date, league
        )
        features.update(
            {
                "date": match.date,
                "league": league,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "result": match.result,
            }
        )
        rows.append(features)

        if position % 5000 == 0:
            print(f"   {position} / {total}...")

    features_df = pd.DataFrame(rows)[ID_COLS + FEATURE_COLS]
    features_df = features_df.dropna(subset=["result"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.output, index=False)

    print(f"\n✅ Built {len(features_df)} rows × {len(FEATURE_COLS)} features")
    print(f"   Saved to {args.output}  ({time.time() - started:.1f}s total)")
    print("\n🎯 Result distribution:")
    print(features_df["result"].value_counts().to_string())
    print("\n🚀 Next:  python train_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

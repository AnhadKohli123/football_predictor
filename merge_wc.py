"""
⚽ Merge international results into the training set.
=====================================================

data/results.csv is the Kaggle "International football results" dump. This
folds selected tournaments into data/all_matches.csv so the model also learns
from national-team football.

    python merge_wc.py                          # merge World Cup into club data
    python merge_wc.py --international-only     # build a dataset with NO API needed
    python merge_wc.py --tournaments "FIFA World Cup" "UEFA Euro" --since 1998

--international-only is the useful one if you have not got an API token yet:
it produces a complete, trainable dataset from the Kaggle CSV alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from features import result_from_goals

KAGGLE_FILE = "data/results.csv"
MATCHES_FILE = "data/all_matches.csv"

# Tournament labels are matched case-insensitively as prefixes, so
# "FIFA World Cup" also catches "FIFA World Cup qualification".
DEFAULT_TOURNAMENTS = ["FIFA World Cup"]


def load_international(path: str, tournaments: list, since: int,
                       include_qualifiers: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    print(f"✅ Loaded {len(frame)} international matches from {path}")

    wanted = [t.lower() for t in tournaments]
    labels = frame["tournament"].astype(str).str.lower()

    if include_qualifiers:
        mask = labels.apply(lambda t: any(t.startswith(w) for w in wanted))
    else:
        mask = labels.isin(wanted)

    selected = frame[mask].copy()
    print(f"   Matching '{', '.join(tournaments)}': {len(selected)}")

    selected = selected[selected["date"].dt.year >= since]
    print(f"   From {since} onwards: {len(selected)}")

    if selected.empty:
        available = frame["tournament"].value_counts().head(12)
        print("\n⚠️  Nothing matched. The most common tournaments in this file are:")
        print(available.to_string())
        return selected

    # The Kaggle file marks matches played at a neutral venue. Those have no
    # real home team, but the schema needs one, so we keep the listed order
    # and let the is_neutral feature carry that information.
    standardised = pd.DataFrame(
        {
            "league": "World Cup",
            "date": selected["date"],
            "matchday": None,
            "home_team": selected["home_team"],
            "away_team": selected["away_team"],
            "home_goals": selected["home_score"],
            "away_goals": selected["away_score"],
            "result": [
                result_from_goals(h, a)
                for h, a in zip(selected["home_score"], selected["away_score"])
            ],
            "season": selected["date"].dt.year.astype(str),
            "stage": selected["tournament"],
        }
    )
    return standardised.dropna(subset=["result"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Merge international results into the dataset.")
    parser.add_argument("--kaggle-file", default=KAGGLE_FILE)
    parser.add_argument("--matches-file", default=MATCHES_FILE)
    parser.add_argument("--tournaments", nargs="+", default=DEFAULT_TOURNAMENTS,
                        help="tournament names to include")
    parser.add_argument("--since", type=int, default=2002, help="earliest year (default: 2002)")
    parser.add_argument("--include-qualifiers", action="store_true",
                        help="also include '<tournament> qualification' matches")
    parser.add_argument("--international-only", action="store_true",
                        help="build the dataset from the Kaggle file alone (no API needed)")
    args = parser.parse_args(argv)

    print("\n⚽ Merging international match data...\n")

    if not Path(args.kaggle_file).exists():
        print(f"❌ Missing {args.kaggle_file}")
        print("   Download 'International football results' from Kaggle into data/")
        return 1

    international = load_international(
        args.kaggle_file, args.tournaments, args.since, args.include_qualifiers
    )
    if international.empty:
        return 1

    club = pd.DataFrame()
    if args.international_only:
        print("\nℹ️  --international-only: ignoring any club data")
    elif Path(args.matches_file).exists():
        club = pd.read_csv(args.matches_file, parse_dates=["date"])
        print(f"\n✅ Loaded {len(club)} club matches from {args.matches_file}")
        club = club[club["league"] != "World Cup"]
    else:
        print(f"\nℹ️  No {args.matches_file} yet — writing an international-only dataset.")
        print("   Run data_fetcher.py later to add club football.")

    combined = pd.concat([club, international], ignore_index=True) if len(club) else international
    combined = combined.dropna(subset=["result"])
    combined = combined.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    Path(args.matches_file).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.matches_file, index=False)

    print(f"\n✅ Combined dataset: {len(combined)} matches")
    print(f"   Club:          {len(club)}")
    print(f"   International: {len(international)}")
    print(f"\n💾 Saved to {args.matches_file}")
    print("\n📈 By competition:")
    print(combined.groupby("league")["result"].count().sort_values(ascending=False).to_string())
    print("\n🎯 Result distribution:")
    print(combined["result"].value_counts().to_string())
    print("\n🚀 Next:\n   1. python build_features.py\n   2. python train_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

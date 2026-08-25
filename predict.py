"""
⚽ Predict upcoming football matches.
=====================================

On-demand: nothing runs in the background. You call it, it fetches only the
fixtures in the window you asked for, reuses anything still fresh in the
cache, and prints predictions.

    python predict.py                          # next 14 days, all leagues
    python predict.py --days 7 --league PL     # one week of Premier League
    python predict.py --match "Arsenal" "Chelsea"   # a single fixture, offline
    python predict.py --force                  # ignore the cache, refetch

Set your API token first (free from https://www.football-data.org/client/register):

    export FOOTBALL_API_KEY='your_token_here'
"""

from __future__ import annotations

import argparse
import logging
import sys

from football_predictor import (
    DEFAULT_DB,
    DEFAULT_ENCODER,
    DEFAULT_HISTORY,
    DEFAULT_MODEL,
    LEAGUES,
    FootballPredictor,
    MissingAPIKey,
    ModelNotTrained,
    Prediction,
)

CONFIDENCE_ICON = {"High": "🔥", "Medium": "✅", "Low": "⚠️ "}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Predict upcoming football matches with your trained model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Leagues: " + ", ".join(LEAGUES),
    )
    parser.add_argument("--days", type=int, default=14,
                        help="how many days ahead to look (default: 14)")
    parser.add_argument("--league", action="append", dest="leagues", metavar="NAME",
                        choices=list(LEAGUES),
                        help="restrict to a league; repeat for several (default: all)")
    parser.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"),
                        help="predict one fixture instead of fetching (works offline)")
    parser.add_argument("--match-league", default="", metavar="NAME",
                        help="competition for --match, e.g. 'Premier League'")
    parser.add_argument("--force", action="store_true",
                        help="ignore cached responses and refetch from the API")
    parser.add_argument("--min-confidence", choices=["Low", "Medium", "High"], default="Low",
                        help="only show predictions at or above this confidence")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model file (default: {DEFAULT_MODEL})")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help=argparse.SUPPRESS)
    parser.add_argument("--history", default=DEFAULT_HISTORY,
                        help=f"historical matches CSV (default: {DEFAULT_HISTORY})")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"cache database (default: {DEFAULT_DB})")
    parser.add_argument("--api-key", default=None,
                        help="API token (otherwise read from FOOTBALL_API_KEY or .env)")
    parser.add_argument("--provider", choices=["api-football", "football-data"], default=None,
                        help="override provider auto-detection")
    parser.add_argument("--csv", default="predictions.csv", help="CSV output path")
    parser.add_argument("--json", default="predictions.json", help="JSON output path")
    parser.add_argument("--no-export", action="store_true", help="print only, write no files")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    return parser


def print_table(predictions: list[Prediction]) -> None:
    home_w = max([len(p.home_team) for p in predictions] + [12])
    away_w = max([len(p.away_team) for p in predictions] + [12])
    header = (
        f"{'DATE':<11} {'HOME':<{home_w}} {'AWAY':<{away_w}} "
        f"{'PREDICTION':<10} {'HOME%':>6} {'DRAW%':>6} {'AWAY%':>6}  CONF"
    )
    print(header)
    print("─" * len(header))

    current_league = None
    for p in predictions:
        if p.league and p.league != current_league:
            current_league = p.league
            print(f"\n  ── {current_league} ──")
        icon = CONFIDENCE_ICON.get(p.confidence, "")
        print(
            f"{p.date:<11} {p.home_team:<{home_w}} {p.away_team:<{away_w}} "
            f"{p.prediction:<10} {p.prob_home_win:>5.1%} {p.prob_draw:>5.1%} "
            f"{p.prob_away_win:>5.1%}  {icon} {p.confidence}"
        )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s" if args.verbose else "%(message)s",
    )
    log = logging.getLogger("predict")

    try:
        with FootballPredictor(
            model_path=args.model,
            encoder_path=args.encoder,
            history_path=args.history,
            db_path=args.db,
            api_key=args.api_key,
            provider=args.provider,
            force=args.force,
        ) as predictor:

            if args.match:
                home, away = args.match
                prediction = predictor.predict_match(
                    home, away, league=args.match_league
                )
                predictions = [prediction]
            else:
                print(f"\n📅 Looking for fixtures in the next {args.days} days...\n")
                fixtures = predictor.fetch_upcoming_fixtures(
                    days=args.days, leagues=args.leagues
                )
                if not fixtures:
                    if predictor.client.failures:
                        # An empty result and a failed request look identical
                        # from here unless we say which one happened.
                        print(
                            f"❌ {predictor.client.failures} request(s) to "
                            f"'{predictor.provider_name}' failed — no fixtures retrieved.\n"
                            "   Check the messages above. Common causes:\n"
                            "     • wrong provider for your key "
                            "(try --provider football-data or --provider api-football)\n"
                            "     • token expired, or the daily quota is used up\n"
                            "     • no internet access from this machine\n"
                            "   Re-run with -v to see the full request details.",
                            file=sys.stderr,
                        )
                        return 5
                    print(
                        "ℹ️  No scheduled fixtures found in that window.\n"
                        "   This is normal mid-summer, or between competition rounds.\n"
                        "   Try a longer window:  python predict.py --days 30"
                    )
                    return 0
                print(f"✅ Found {len(fixtures)} fixtures — predicting...\n")

                unknown = predictor.unknown_teams(fixtures)
                if unknown:
                    # Predictions for these rest entirely on neutral priors,
                    # so they carry far less information than they look like.
                    preview = ", ".join(unknown[:6])
                    more = f" (+{len(unknown) - 6} more)" if len(unknown) > 6 else ""
                    print(f"⚠️  No history for: {preview}{more}")
                    print("   Their predictions fall back to league-average priors.")
                    print("   Fetch more history with data_fetcher.py to fix this.\n")

                predictions = predictor.predict_fixtures(fixtures)

            ranking = {"Low": 0, "Medium": 1, "High": 2}
            floor = ranking[args.min_confidence]
            shown = [p for p in predictions if ranking[p.confidence] >= floor]

            if not shown:
                print(f"ℹ️  No predictions at '{args.min_confidence}' confidence or above.")
                return 0

            print_table(shown)

            hidden = len(predictions) - len(shown)
            print(f"\n📊 {len(shown)} prediction(s)" + (f", {hidden} below the confidence floor" if hidden else ""))
            if predictor.client.requests_made:
                print(f"🌐 {predictor.client.requests_made} API request(s) made this run")
            else:
                print("💾 Served entirely from cache — no API requests made")

            if not args.no_export:
                predictor.export(predictions, csv_path=args.csv, json_path=args.json)
                print(f"💾 Saved to {args.csv} and {args.json}")

            print(
                "\n   🔥 High = strong signal   ✅ Medium = reasonable   ⚠️  Low = uncertain\n"
                "   Football is genuinely hard to predict — treat these as odds, not answers."
            )
        return 0

    except MissingAPIKey as exc:
        print(f"\n🔑 {exc}\n", file=sys.stderr)
        return 2
    except ModelNotTrained as exc:
        print(f"\n❌ {exc}\n", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"\n❌ {exc}\n", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

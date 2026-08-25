"""
⚽ Web frontend for the football predictor.
===========================================

    pip install flask
    python app.py

Then open http://127.0.0.1:5000

Everything the CLI does is available here too. The single-match predictor
works entirely offline; the fixtures view needs an API token in .env.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from football_predictor import (
    DEFAULT_DB,
    DEFAULT_ENCODER,
    DEFAULT_HISTORY,
    DEFAULT_MODEL,
    FootballPredictor,
    MissingAPIKey,
    ModelNotTrained,
)
from providers import LEAGUE_NAMES

log = logging.getLogger("app")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Set by main(); the predictor itself is built per request so a retrained
# model is picked up without restarting the server.
CONFIG = {
    "model": DEFAULT_MODEL,
    "encoder": DEFAULT_ENCODER,
    "history": DEFAULT_HISTORY,
    "db": DEFAULT_DB,
    "provider": None,
}


def make_predictor(force: bool = False) -> FootballPredictor:
    return FootballPredictor(
        model_path=CONFIG["model"],
        encoder_path=CONFIG["encoder"],
        history_path=CONFIG["history"],
        db_path=CONFIG["db"],
        provider=CONFIG["provider"],
        force=force,
    )


def error(message: str, status: int = 400, **extra):
    payload = {"ok": False, "error": message}
    payload.update(extra)
    return jsonify(payload), status


# ============================================================
# Pages
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# API
# ============================================================

@app.route("/api/status")
def api_status():
    """What the app can currently do — drives the banners in the UI."""
    status = {
        "ok": True,
        "model_ready": Path(CONFIG["model"]).exists() and Path(CONFIG["encoder"]).exists(),
        "history_ready": Path(CONFIG["history"]).exists(),
        "api_key_set": False,
        "provider": None,
        "leagues": [{"key": k, "name": v} for k, v in LEAGUE_NAMES.items()],
        "metrics": None,
        "teams": 0,
        "matches": 0,
        "date_range": None,
    }

    metrics_file = Path(CONFIG["model"]).parent / "metrics.json"
    if metrics_file.exists():
        try:
            status["metrics"] = json.loads(metrics_file.read_text())
        except json.JSONDecodeError:
            pass

    try:
        with make_predictor() as predictor:
            status["api_key_set"] = bool(predictor.api_key)
            status["provider"] = predictor.provider_name
            if status["history_ready"]:
                history = predictor.history
                status["teams"] = len(history.teams())
                status["matches"] = history.n_matches
                if history.min_date is not None:
                    status["date_range"] = [
                        str(history.min_date.date()),
                        str(history.max_date.date()),
                    ]
    except Exception as exc:  # a broken setup must still render the page
        log.warning("status check failed: %s", exc)
        status["warning"] = str(exc)

    return jsonify(status)


@app.route("/api/teams")
def api_teams():
    """Every team the model has history for — powers the pickers."""
    try:
        with make_predictor() as predictor:
            return jsonify({"ok": True, "teams": predictor.history.teams()})
    except FileNotFoundError as exc:
        return error(str(exc), 503)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict one fixture. No network access required."""
    body = request.get_json(silent=True) or {}
    home = (body.get("home") or "").strip()
    away = (body.get("away") or "").strip()
    league = (body.get("league") or "").strip()
    date = (body.get("date") or "").strip() or None

    if not home or not away:
        return error("Pick both a home and an away team.")
    if home == away:
        return error("A team cannot play itself.")

    try:
        with make_predictor() as predictor:
            history = predictor.history
            unknown = [t for t in (home, away) if not history.has_team(t)]
            prediction = predictor.predict_match(home, away, date, league)
            payload = prediction.as_row()
            payload["unknown_teams"] = unknown
            # Surfaced in the UI: a prediction resting on priors is far less
            # informative than the percentages alone suggest.
            payload["reliable"] = not unknown
            return jsonify({"ok": True, "prediction": payload})
    except ModelNotTrained as exc:
        return error(str(exc), 503)
    except FileNotFoundError as exc:
        return error(str(exc), 503)
    except Exception as exc:
        log.exception("prediction failed")
        return error(f"Prediction failed: {exc}", 500)


@app.route("/api/fixtures")
def api_fixtures():
    """Fetch upcoming fixtures and predict them. Needs an API token."""
    try:
        days = max(1, min(90, int(request.args.get("days", 14))))
    except ValueError:
        return error("'days' must be a number.")

    leagues = [l for l in request.args.getlist("league") if l in LEAGUE_NAMES] or None
    force = request.args.get("force") == "true"

    try:
        with make_predictor(force=force) as predictor:
            fixtures = predictor.fetch_upcoming_fixtures(days=days, leagues=leagues)

            if not fixtures:
                if predictor.client.failures:
                    return error(
                        f"{predictor.client.failures} request(s) to "
                        f"'{predictor.provider_name}' failed. Check that your token is "
                        "valid and matches the provider, and that the daily quota is "
                        "not used up.",
                        502,
                        provider=predictor.provider_name,
                    )
                return jsonify({
                    "ok": True, "fixtures": [], "requests_made": 0,
                    "message": "No scheduled fixtures in that window. "
                               "This is normal between competition rounds — try more days.",
                })

            predictions = predictor.predict_fixtures(fixtures)
            unknown = set(predictor.unknown_teams(fixtures))
            rows = []
            for prediction in predictions:
                row = prediction.as_row()
                row["reliable"] = not (
                    row["home_team"] in unknown or row["away_team"] in unknown
                )
                rows.append(row)

            return jsonify({
                "ok": True,
                "fixtures": rows,
                "requests_made": predictor.client.requests_made,
                "provider": predictor.provider_name,
                "unknown_teams": sorted(unknown),
            })

    except MissingAPIKey as exc:
        return error(str(exc), 401)
    except ModelNotTrained as exc:
        return error(str(exc), 503)
    except FileNotFoundError as exc:
        return error(str(exc), 503)
    except Exception as exc:
        log.exception("fixture fetch failed")
        return error(f"Could not load fixtures: {exc}", 500)


@app.route("/api/history")
def api_history():
    """Recent predictions logged by the CLI and by this app."""
    try:
        limit = max(1, min(200, int(request.args.get("limit", 50))))
    except ValueError:
        limit = 50

    try:
        with make_predictor() as predictor:
            rows = predictor.cache.conn.execute(
                "SELECT home_team, away_team, league, utc_date AS date, prediction,"
                "       p_home, p_draw, p_away, predicted_at, actual"
                "  FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return jsonify({"ok": True, "predictions": [dict(r) for r in rows]})
    except Exception as exc:
        log.exception("history read failed")
        return error(f"Could not read prediction history: {exc}", 500)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the web frontend.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default: 127.0.0.1, local only)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="auto-reload on edits")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--history", default=DEFAULT_HISTORY)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--provider", choices=["api-football", "football-data"], default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    CONFIG.update({
        "model": args.model, "encoder": args.encoder,
        "history": args.history, "db": args.db, "provider": args.provider,
    })

    if not Path(args.model).exists():
        print("⚠️  No trained model found — the page will load but cannot predict.")
        print("   Train one first:  python build_features.py && python train_model.py\n")

    print(f"\n⚽ Football Predictor running at  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())

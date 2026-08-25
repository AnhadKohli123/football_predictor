"""
Core prediction engine.
=======================

Wraps three things behind one class:

  * a rate-limited, cached fixture provider (see providers.py)
  * an on-demand fixture fetcher — only the next N days, nothing more
  * the trained model, fed through the shared feature engine in features.py

Nothing here runs on a schedule. You call it, it fetches only what it needs,
and reuses anything it already has that is still fresh.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from features import FEATURE_COLS, MatchHistory, build_features_for_match, to_model_frame
from providers import LEAGUE_NAMES, LEAGUE_TABLE, detect_provider, get_provider

log = logging.getLogger("football_predictor")

# ============================================================
# Configuration
# ============================================================

#: Environment variables checked, in order, for the API token.
API_KEY_ENV_VARS = ("FOOTBALL_API_KEY", "FOOTBALL_DATA_API_TOKEN", "API_FOOTBALL_KEY")
PROVIDER_ENV_VAR = "FOOTBALL_API_PROVIDER"

#: Short CLI names -> display name. Provider-specific ids live in providers.py.
LEAGUES: Dict[str, str] = LEAGUE_NAMES

DEFAULT_DB = "football_cache.db"
DEFAULT_MODEL = "models/predictor.pkl"
DEFAULT_ENCODER = "models/label_encoder.pkl"
DEFAULT_HISTORY = "data/all_matches.csv"

FIXTURES_TTL_SECONDS = 6 * 3600
GENERIC_TTL_SECONDS = 2 * 3600

PLACEHOLDER_KEYS = {"YOUR_TOKEN_HERE", "your_token_here", "changeme", ""}


class MissingAPIKey(RuntimeError):
    """Raised when a network call is needed but no token is configured."""


class ModelNotTrained(RuntimeError):
    """Raised when the model artefacts are missing."""


def _read_env_file(path: Path = Path(".env")) -> Dict[str, str]:
    """Minimal .env reader — no dependency on python-dotenv."""
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip().strip("'\"")
    return values


def resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """
    Find the API token: explicit argument, then environment, then .env.

    Returns None when nothing is configured. That is only fatal for calls
    that actually hit the network — offline prediction works without a key.
    """
    if explicit and explicit.strip() not in PLACEHOLDER_KEYS:
        return explicit.strip()

    for var in API_KEY_ENV_VARS:
        value = (os.environ.get(var) or "").strip()
        if value and value not in PLACEHOLDER_KEYS:
            return value

    env_values = _read_env_file()
    for var in API_KEY_ENV_VARS:
        value = (env_values.get(var) or "").strip()
        if value and value not in PLACEHOLDER_KEYS:
            return value
    return None


def resolve_provider(explicit: Optional[str] = None) -> Optional[str]:
    """Provider override from the CLI, environment, or .env — else None (auto)."""
    if explicit:
        return explicit
    value = (os.environ.get(PROVIDER_ENV_VAR) or "").strip()
    if value:
        return value
    return _read_env_file().get(PROVIDER_ENV_VAR) or None


API_KEY_HELP = f"""
No football API token found.

  1. Get a free token from either provider:
       https://www.football-data.org/client/register   (32-character key)
       https://dashboard.api-football.com/register      (40-character key)

  2. Make it available, any one of these ways:

       export {API_KEY_ENV_VARS[0]}='your_token_here'

     ...or put it in a file named .env next to this script:

       {API_KEY_ENV_VARS[0]}=your_token_here

     ...or pass it directly:

       python predict.py --api-key your_token_here

The provider is auto-detected from the key's format. Override it with
--provider api-football  or  --provider football-data  if the guess is wrong.

.env is listed in .gitignore, so it will not be committed.
""".strip()


# ============================================================
# Cache
# ============================================================

class Cache:
    """SQLite-backed cache. One file, no server, safe to delete at any time."""

    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        parent = Path(path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key  TEXT PRIMARY KEY,
                url        TEXT NOT NULL,
                payload    TEXT NOT NULL,
                fetched_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS upcoming_fixtures (
                fixture_id INTEGER PRIMARY KEY,
                league     TEXT NOT NULL,
                utc_date   TEXT NOT NULL,
                home_team  TEXT NOT NULL,
                away_team  TEXT NOT NULL,
                matchday   TEXT,
                status     TEXT,
                fetched_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fixtures_date
                ON upcoming_fixtures (utc_date);

            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id   INTEGER,
                league       TEXT,
                utc_date     TEXT,
                home_team    TEXT,
                away_team    TEXT,
                prediction   TEXT,
                p_home       REAL,
                p_draw       REAL,
                p_away       REAL,
                predicted_at REAL,
                actual       TEXT
            );
            """
        )
        self.conn.commit()

    def get_json(self, cache_key: str, ttl: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT payload, fetched_at FROM api_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None or time.time() - row["fetched_at"] > ttl:
            return None
        try:
            return json.loads(row["payload"])
        except json.JSONDecodeError:
            return None

    def put_json(self, cache_key: str, url: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO api_cache (cache_key, url, payload, fetched_at)"
            " VALUES (?, ?, ?, ?)",
            (cache_key, url, json.dumps(payload), time.time()),
        )
        self.conn.commit()

    def save_fixtures(self, fixtures: Iterable[dict]) -> None:
        now = time.time()
        rows = [
            (
                f.get("fixture_id"), f["league"], f["utc_date"], f["home_team"],
                f["away_team"], str(f.get("matchday") or ""), f.get("status"), now,
            )
            for f in fixtures
            if f.get("fixture_id") is not None
        ]
        if not rows:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO upcoming_fixtures"
            " (fixture_id, league, utc_date, home_team, away_team, matchday, status, fetched_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def log_predictions(self, rows: Iterable[dict]) -> None:
        now = time.time()
        self.conn.executemany(
            "INSERT INTO predictions"
            " (fixture_id, league, utc_date, home_team, away_team, prediction,"
            "  p_home, p_draw, p_away, predicted_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r.get("fixture_id"), r.get("league"), r.get("date"), r.get("home_team"),
                    r.get("away_team"), r.get("prediction"), r.get("prob_home_win"),
                    r.get("prob_draw"), r.get("prob_away_win"), now,
                )
                for r in rows
            ],
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ============================================================
# Predictions
# ============================================================

PRETTY_RESULT = {"HOME_WIN": "Home Win", "DRAW": "Draw", "AWAY_WIN": "Away Win"}


def confidence_label(prob: float) -> str:
    if prob >= 0.60:
        return "High"
    if prob >= 0.45:
        return "Medium"
    return "Low"


@dataclass
class Prediction:
    league: str
    date: str
    home_team: str
    away_team: str
    prediction: str
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    confidence: str
    fixture_id: Optional[int] = None

    def as_row(self) -> dict:
        return {
            "date": self.date,
            "league": self.league,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "prediction": self.prediction,
            "prob_home_win": round(self.prob_home_win, 4),
            "prob_draw": round(self.prob_draw, 4),
            "prob_away_win": round(self.prob_away_win, 4),
            "confidence": self.confidence,
            "fixture_id": self.fixture_id,
        }


# ============================================================
# Predictor
# ============================================================

class FootballPredictor:
    """The object predict.py drives."""

    def __init__(self, model_path: str = DEFAULT_MODEL, encoder_path: str = DEFAULT_ENCODER,
                 history_path: str = DEFAULT_HISTORY, db_path: str = DEFAULT_DB,
                 api_key: Optional[str] = None, provider: Optional[str] = None,
                 force: bool = False):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.history_path = history_path

        self.api_key = resolve_api_key(api_key)
        self.provider_name = resolve_provider(provider) or detect_provider(self.api_key)

        self.cache = Cache(db_path)
        self.client = get_provider(
            self.api_key, self.cache, force=force, provider=self.provider_name
        )

        self._model = None
        self._encoder = None
        self._history: Optional[MatchHistory] = None

    # -- lazy loading --------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def encoder(self):
        if self._encoder is None:
            self._load_model()
        return self._encoder

    def _load_model(self) -> None:
        import joblib  # lazy, so --help works even without sklearn present

        if not Path(self.model_path).exists() or not Path(self.encoder_path).exists():
            raise ModelNotTrained(
                f"No trained model at '{self.model_path}'.\n"
                "Build one first:\n"
                "    python merge_wc.py --international-only   # or: python data_fetcher.py\n"
                "    python build_features.py\n"
                "    python train_model.py"
            )
        self._model = joblib.load(self.model_path)
        self._encoder = joblib.load(self.encoder_path)

        trained_on = getattr(self._model, "feature_names_in_", None)
        if trained_on is not None and list(trained_on) != FEATURE_COLS:
            log.warning(
                "this model was trained on a different feature set than features.py "
                "defines — re-run train_model.py so the two agree"
            )

    @property
    def history(self) -> MatchHistory:
        if self._history is None:
            path = Path(self.history_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"No match history at '{path}'.\n"
                    "Build it with:\n"
                    "    python merge_wc.py --international-only   # offline, no token\n"
                    "    python data_fetcher.py                    # club football, needs a token"
                )
            self._history = MatchHistory(pd.read_csv(path))
            log.info(
                "history: %s matches (%s → %s)",
                self._history.n_matches,
                self._history.min_date.date() if self._history.min_date is not None else "?",
                self._history.max_date.date() if self._history.max_date is not None else "?",
            )
        return self._history

    # -- fixtures ------------------------------------------------------

    def fetch_upcoming_fixtures(self, days: int = 14,
                                leagues: Optional[List[str]] = None) -> List[dict]:
        """
        Fetch scheduled matches within the next `days` days.

        One request per competition, cached for six hours, so re-running
        within the same afternoon costs no API quota at all.
        """
        keys = [k for k in (leagues or list(LEAGUE_TABLE)) if k in LEAGUE_TABLE]
        unknown = set(leagues or []) - set(LEAGUE_TABLE)
        for key in sorted(unknown):
            log.warning("unknown league '%s' — skipping", key)

        fixtures = self.client.fetch_fixtures(days=days, leagues=keys)
        fixtures.sort(key=lambda f: (f.get("utc_date") or "", f.get("league") or ""))
        if fixtures:
            self.cache.save_fixtures(fixtures)
        return fixtures

    # -- prediction ----------------------------------------------------

    def predict_match(self, home_team: str, away_team: str, date=None,
                      league: str = "") -> Prediction:
        """Predict a single fixture. Needs no network access."""
        when = pd.Timestamp(date) if date is not None else pd.Timestamp(datetime.now().date())
        features = build_features_for_match(self.history, home_team, away_team, when, league)
        frame = to_model_frame([features])

        probabilities = self.model.predict_proba(frame)[0]
        by_label = dict(zip(self.encoder.classes_, probabilities))
        best = max(by_label, key=by_label.get)

        return Prediction(
            league=league,
            date=str(when.date()),
            home_team=home_team,
            away_team=away_team,
            prediction=PRETTY_RESULT.get(best, best),
            prob_home_win=float(by_label.get("HOME_WIN", 0.0)),
            prob_draw=float(by_label.get("DRAW", 0.0)),
            prob_away_win=float(by_label.get("AWAY_WIN", 0.0)),
            confidence=confidence_label(float(max(probabilities))),
        )

    def predict_fixtures(self, fixtures: List[dict]) -> List[Prediction]:
        """Predict a batch of fixtures with a single vectorised model call."""
        if not fixtures:
            return []

        history = self.history
        rows = [
            build_features_for_match(
                history, f["home_team"], f["away_team"], pd.Timestamp(f["date"]), f["league"]
            )
            for f in fixtures
        ]
        probabilities = self.model.predict_proba(to_model_frame(rows))
        labels = list(self.encoder.classes_)

        predictions = []
        for fixture, probs in zip(fixtures, probabilities):
            by_label = dict(zip(labels, probs))
            best = max(by_label, key=by_label.get)
            predictions.append(
                Prediction(
                    league=fixture["league"],
                    date=fixture["date"],
                    home_team=fixture["home_team"],
                    away_team=fixture["away_team"],
                    prediction=PRETTY_RESULT.get(best, best),
                    prob_home_win=float(by_label.get("HOME_WIN", 0.0)),
                    prob_draw=float(by_label.get("DRAW", 0.0)),
                    prob_away_win=float(by_label.get("AWAY_WIN", 0.0)),
                    confidence=confidence_label(float(max(probs))),
                    fixture_id=fixture.get("fixture_id"),
                )
            )
        return predictions

    def unknown_teams(self, fixtures: List[dict]) -> List[str]:
        """Fixture teams with no history — their predictions rest on priors alone."""
        history = self.history
        names = {f["home_team"] for f in fixtures} | {f["away_team"] for f in fixtures}
        return sorted(n for n in names if not history.has_team(n))

    # -- output --------------------------------------------------------

    def export(self, predictions: List[Prediction], csv_path: str = "predictions.csv",
               json_path: str = "predictions.json") -> None:
        rows = [p.as_row() for p in predictions]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        Path(json_path).write_text(json.dumps(rows, indent=2))
        self.cache.log_predictions(rows)

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> "FootballPredictor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

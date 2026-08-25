"""
Canonical feature engineering — the single source of truth.
===========================================================

Both training (build_features.py) and inference (football_predictor.py /
predict.py) import from this module. That matters: if the two ever built
features differently, the model would be fed vectors that mean something
different from what it learned, and the predictions would be quietly wrong.

Everything here is "as-of" — a feature for a match played on date D is
computed only from matches that finished strictly BEFORE D. That is what
keeps the training set free of leakage.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# ============================================================
# Constants shared by every stage of the pipeline
# ============================================================

FORM_WINDOW = 5  # how many recent matches count as "form"

LEAGUE_MAP: Dict[str, int] = {
    "Premier League":   0,
    "La Liga":          1,
    "Bundesliga":       2,
    "Serie A":          3,
    "Ligue 1":          4,
    "Eredivisie":       5,
    "Primeira Liga":    6,
    "Champions League": 7,
    "World Cup":        8,
}

# Order matters — the model is trained on exactly this column order.
FEATURE_COLS: List[str] = [
    "home_form_points",
    "home_form_goals_scored",
    "home_form_goals_conceded",
    "home_form_goal_diff",
    "home_form_wins",
    "home_form_draws",
    "home_form_losses",
    "away_form_points",
    "away_form_goals_scored",
    "away_form_goals_conceded",
    "away_form_goal_diff",
    "away_form_wins",
    "away_form_draws",
    "away_form_losses",
    "home_venue_win_rate",
    "home_venue_goals_scored",
    "home_venue_goals_conceded",
    "away_travel_win_rate",
    "away_travel_goals_scored",
    "away_travel_goals_conceded",
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "diff_form_points",
    "diff_form_goal_diff",
    "diff_attack",
    "diff_defence",
    "league_code",
    "is_neutral",
]

# Columns that identify a match but are never fed to the model.
ID_COLS: List[str] = ["date", "league", "home_team", "away_team", "result"]

RESULT_HOME_WIN = "HOME_WIN"
RESULT_AWAY_WIN = "AWAY_WIN"
RESULT_DRAW = "DRAW"

# Fallbacks for teams with no prior history (promoted sides, first match of
# the dataset, a nation that has not played since the last World Cup...).
# These are league-average-ish priors rather than zeros, so a debut team is
# treated as "average and unknown" instead of "catastrophically bad".
NEUTRAL_FORM = {
    "form_points": 1.0,
    "form_goals_scored": 1.2,
    "form_goals_conceded": 1.2,
    "form_goal_diff": 0.0,
    "form_wins": 0.33,
    "form_draws": 0.33,
    "form_losses": 0.33,
}
NEUTRAL_HOME = {
    "home_win_rate": 0.45,
    "home_goals_scored": 1.3,
    "home_goals_conceded": 1.1,
}
NEUTRAL_AWAY = {
    "away_win_rate": 0.30,
    "away_goals_scored": 1.1,
    "away_goals_conceded": 1.3,
}
NEUTRAL_H2H = {
    "h2h_home_wins": 0.33,
    "h2h_away_wins": 0.33,
    "h2h_draws": 0.33,
}


# ============================================================
# Team name matching
# ============================================================

# Every data source spells clubs differently: openfootball says "Arsenal FC",
# API-Football says "Arsenal", football-data.org says "Arsenal FC", and a
# human types "arsenal". Without normalisation, a fixture would look like a
# brand-new team and fall back to priors — silently making every prediction
# useless. Names are therefore matched on a normalised key, while the original
# spelling is kept for display.

#: Club-type words that carry no identifying information.
CLUB_TOKENS = {
    "fc", "afc", "cf", "ac", "as", "sc", "ss", "ssc", "sv", "tsv", "tsg",
    "vfl", "vfb", "bsc", "fsv", "spvgg", "rc", "rcd", "cd", "ud", "sd", "cp",
    "aс", "us", "usc", "asd", "calcio", "club", "de", "futbol", "football",
    "borussia", "deportivo", "real" if False else "", "the",
}
CLUB_TOKENS.discard("")

#: Names that normalisation alone will not reconcile.
TEAM_ALIASES = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "brighton": "brighton hove albion",
    "brighton and hove albion": "brighton hove albion",
    "west brom": "west bromwich albion",
    "newcastle": "newcastle united",
    "leeds": "leeds united",
    "west ham": "west ham united",
    "nottingham forest": "nottingham forest",
    "notts forest": "nottingham forest",
    "sheffield utd": "sheffield united",
    "inter": "internazionale",
    "inter milan": "internazionale",
    "ac milan": "milan",
    "atletico madrid": "atletico de madrid",
    "atl madrid": "atletico de madrid",
    "athletic bilbao": "athletic club",
    "barcelona": "barcelona",
    "fc barcelona": "barcelona",
    "bayern": "bayern munchen",
    "bayern munich": "bayern munchen",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "dortmund": "borussia dortmund",
    "monchengladbach": "borussia monchengladbach",
    "leverkusen": "bayer 04 leverkusen",
    "bayer leverkusen": "bayer 04 leverkusen",
    "hoffenheim": "1899 hoffenheim",
    "koln": "1 fc koln",
    "cologne": "1 fc koln",
    "psv": "psv eindhoven",
    "sporting": "sporting cp",
    "sporting lisbon": "sporting cp",
    "porto": "fc porto",
    "benfica": "sl benfica",
    "roma": "as roma",
    "napoli": "ssc napoli",
    "juventus": "juventus",
    "lazio": "ss lazio",
    "marseille": "olympique marseille",
    "lyon": "olympique lyonnais",
}

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def team_key(name: str) -> str:
    """
    A normalised key for matching a team across data sources.

    "Arsenal FC", "arsenal", and "Arsenal F.C." all collapse to "arsenal".
    Returns an empty string for empty input.
    """
    if not name:
        return ""

    # Strip accents: "Köln" -> "Koln", "Atlético" -> "Atletico".
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ")
    text = _NON_ALNUM.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()

    if text in TEAM_ALIASES:
        text = TEAM_ALIASES[text]

    # Drop club-type words, but never every word — "AC Milan" must not become
    # empty, and a name made only of such tokens keeps its original form.
    words = [w for w in text.split() if w not in CLUB_TOKENS]
    if words:
        text = " ".join(words)

    if text in TEAM_ALIASES:
        text = TEAM_ALIASES[text]

    return text


def result_from_goals(home_goals, away_goals) -> Optional[str]:
    """Result from the HOME team's perspective."""
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    if home_goals > away_goals:
        return RESULT_HOME_WIN
    if home_goals < away_goals:
        return RESULT_AWAY_WIN
    return RESULT_DRAW


# ============================================================
# Indexed match history
# ============================================================

def _as_nanos(value) -> int:
    """Normalise any date-ish value to an int for fast comparison."""
    return pd.Timestamp(value).to_datetime64().astype("datetime64[ns]").astype("int64")


class MatchHistory:
    """
    An immutable, date-indexed view over finished matches.

    The naive approach — re-filtering the whole DataFrame for every lookup —
    is O(n) per call and there are five calls per match, so building a
    training set is O(n^2). On the ~50k match dataset in data/ that is a few
    billion row comparisons and takes hours.

    Instead we bucket matches by team (and by opponent pair) once, keep each
    bucket sorted by date, and binary-search it. Building the same training
    set becomes O(n log n) and runs in seconds.
    """

    def __init__(self, df: pd.DataFrame):
        required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"MatchHistory needs columns {sorted(missing)}")

        frame = df.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
        frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
        frame = frame.dropna(subset=["date", "home_goals", "away_goals"])
        frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)

        # Every index below is keyed by team_key(name), never the raw name,
        # so "Arsenal" and "Arsenal FC" land in the same bucket.
        self._display: Dict[str, str] = {}

        # team -> parallel lists of (date, goals_for, goals_against, points)
        self._team_dates: Dict[str, List[int]] = {}
        self._team_rows: Dict[str, List[Tuple[float, float, int]]] = {}
        # team -> matches played at home / away only
        self._home_dates: Dict[str, List[int]] = {}
        self._home_rows: Dict[str, List[Tuple[float, float, bool]]] = {}
        self._away_dates: Dict[str, List[int]] = {}
        self._away_rows: Dict[str, List[Tuple[float, float, bool]]] = {}
        # unordered team pair -> (date, home_team, result)
        self._pair_dates: Dict[Tuple[str, str], List[int]] = {}
        self._pair_rows: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

        self.n_matches = len(frame)
        self.min_date = frame["date"].min() if self.n_matches else None
        self.max_date = frame["date"].max() if self.n_matches else None

        for row in frame.itertuples(index=False):
            stamp = _as_nanos(row.date)
            home, away = team_key(row.home_team), team_key(row.away_team)
            if not home or not away:
                continue
            self._display.setdefault(home, str(row.home_team))
            self._display.setdefault(away, str(row.away_team))
            hg, ag = float(row.home_goals), float(row.away_goals)
            result = result_from_goals(hg, ag)
            if result is None:
                continue

            home_pts = 3 if result == RESULT_HOME_WIN else (1 if result == RESULT_DRAW else 0)
            away_pts = 3 if result == RESULT_AWAY_WIN else (1 if result == RESULT_DRAW else 0)

            self._team_dates.setdefault(home, []).append(stamp)
            self._team_rows.setdefault(home, []).append((hg, ag, home_pts))
            self._team_dates.setdefault(away, []).append(stamp)
            self._team_rows.setdefault(away, []).append((ag, hg, away_pts))

            self._home_dates.setdefault(home, []).append(stamp)
            self._home_rows.setdefault(home, []).append((hg, ag, result == RESULT_HOME_WIN))
            self._away_dates.setdefault(away, []).append(stamp)
            self._away_rows.setdefault(away, []).append((ag, hg, result == RESULT_AWAY_WIN))

            pair = (home, away) if home <= away else (away, home)
            self._pair_dates.setdefault(pair, []).append(stamp)
            self._pair_rows.setdefault(pair, []).append((home, result))

    # -- internal ------------------------------------------------------

    @staticmethod
    def _window(dates: Sequence[int], rows: Sequence, before: int, n: int) -> Sequence:
        """The last `n` rows strictly before `before`."""
        cut = bisect.bisect_left(dates, before)
        if cut == 0:
            return ()
        return rows[max(0, cut - n):cut]

    # -- feature blocks ------------------------------------------------

    def team_form(self, team: str, before, n: int = FORM_WINDOW) -> Optional[dict]:
        """Overall form over the last `n` matches, home or away."""
        team = team_key(team)
        rows = self._window(
            self._team_dates.get(team, ()), self._team_rows.get(team, ()), _as_nanos(before), n
        )
        if not rows:
            return None
        played = len(rows)
        scored = sum(r[0] for r in rows)
        conceded = sum(r[1] for r in rows)
        points = sum(r[2] for r in rows)
        wins = sum(1 for r in rows if r[2] == 3)
        draws = sum(1 for r in rows if r[2] == 1)
        return {
            "form_points": points / played,
            "form_goals_scored": scored / played,
            "form_goals_conceded": conceded / played,
            "form_goal_diff": (scored - conceded) / played,
            "form_wins": wins / played,
            "form_draws": draws / played,
            "form_losses": (played - wins - draws) / played,
        }

    def home_form(self, team: str, before, n: int = FORM_WINDOW) -> Optional[dict]:
        """Form in the last `n` matches played at home."""
        team = team_key(team)
        rows = self._window(
            self._home_dates.get(team, ()), self._home_rows.get(team, ()), _as_nanos(before), n
        )
        if not rows:
            return None
        played = len(rows)
        return {
            "home_win_rate": sum(1 for r in rows if r[2]) / played,
            "home_goals_scored": sum(r[0] for r in rows) / played,
            "home_goals_conceded": sum(r[1] for r in rows) / played,
        }

    def away_form(self, team: str, before, n: int = FORM_WINDOW) -> Optional[dict]:
        """Form in the last `n` matches played away from home."""
        team = team_key(team)
        rows = self._window(
            self._away_dates.get(team, ()), self._away_rows.get(team, ()), _as_nanos(before), n
        )
        if not rows:
            return None
        played = len(rows)
        return {
            "away_win_rate": sum(1 for r in rows if r[2]) / played,
            "away_goals_scored": sum(r[0] for r in rows) / played,
            "away_goals_conceded": sum(r[1] for r in rows) / played,
        }

    def head_to_head(self, home_team: str, away_team: str, before,
                     n: int = FORM_WINDOW) -> Optional[dict]:
        """
        Recent meetings, oriented to THIS fixture's home team.

        A win for `home_team` counts as a home win even if it happened at the
        other team's ground, so the feature always means "the team hosting
        today has historically beaten this opponent".
        """
        home_team, away_team = team_key(home_team), team_key(away_team)
        pair = (home_team, away_team) if home_team <= away_team else (away_team, home_team)
        rows = self._window(
            self._pair_dates.get(pair, ()), self._pair_rows.get(pair, ()), _as_nanos(before), n
        )
        if not rows:
            return None
        hw = aw = dr = 0
        for past_home, result in rows:
            if result == RESULT_DRAW:
                dr += 1
            elif (result == RESULT_HOME_WIN) == (past_home == home_team):
                hw += 1
            else:
                aw += 1
        played = len(rows)
        return {
            "h2h_home_wins": hw / played,
            "h2h_away_wins": aw / played,
            "h2h_draws": dr / played,
        }

    def has_team(self, team: str) -> bool:
        return team_key(team) in self._team_dates

    def teams(self) -> List[str]:
        """Display names, as they were spelled in the source data."""
        return sorted(self._display[k] for k in self._team_dates if k in self._display)

    def display_name(self, team: str) -> str:
        """The stored spelling for a team, or the input if it is unknown."""
        return self._display.get(team_key(team), team)


# ============================================================
# The feature vector
# ============================================================

def build_features_for_match(history: MatchHistory, home_team: str, away_team: str,
                             date, league: str = "") -> Dict[str, float]:
    """Build one model-ready feature dict. Keys are exactly FEATURE_COLS."""
    hf = history.team_form(home_team, date) or NEUTRAL_FORM
    af = history.team_form(away_team, date) or NEUTRAL_FORM
    hh = history.home_form(home_team, date) or NEUTRAL_HOME
    aa = history.away_form(away_team, date) or NEUTRAL_AWAY
    h2 = history.head_to_head(home_team, away_team, date) or NEUTRAL_H2H

    return {
        # Home team overall form
        "home_form_points": hf["form_points"],
        "home_form_goals_scored": hf["form_goals_scored"],
        "home_form_goals_conceded": hf["form_goals_conceded"],
        "home_form_goal_diff": hf["form_goal_diff"],
        "home_form_wins": hf["form_wins"],
        "home_form_draws": hf["form_draws"],
        "home_form_losses": hf["form_losses"],
        # Away team overall form
        "away_form_points": af["form_points"],
        "away_form_goals_scored": af["form_goals_scored"],
        "away_form_goals_conceded": af["form_goals_conceded"],
        "away_form_goal_diff": af["form_goal_diff"],
        "away_form_wins": af["form_wins"],
        "away_form_draws": af["form_draws"],
        "away_form_losses": af["form_losses"],
        # Venue-specific records
        "home_venue_win_rate": hh["home_win_rate"],
        "home_venue_goals_scored": hh["home_goals_scored"],
        "home_venue_goals_conceded": hh["home_goals_conceded"],
        "away_travel_win_rate": aa["away_win_rate"],
        "away_travel_goals_scored": aa["away_goals_scored"],
        "away_travel_goals_conceded": aa["away_goals_conceded"],
        # Head to head
        "h2h_home_wins": h2["h2h_home_wins"],
        "h2h_away_wins": h2["h2h_away_wins"],
        "h2h_draws": h2["h2h_draws"],
        # Differences — usually the most informative features
        "diff_form_points": hf["form_points"] - af["form_points"],
        "diff_form_goal_diff": hf["form_goal_diff"] - af["form_goal_diff"],
        "diff_attack": hf["form_goals_scored"] - af["form_goals_conceded"],
        "diff_defence": af["form_goals_scored"] - hf["form_goals_conceded"],
        # Context
        "league_code": LEAGUE_MAP.get(league, -1),
        "is_neutral": 1 if league == "World Cup" else 0,
    }


def to_model_frame(feature_dicts) -> pd.DataFrame:
    """Stack feature dicts into a DataFrame with columns in FEATURE_COLS order."""
    frame = pd.DataFrame(list(feature_dicts))
    for col in FEATURE_COLS:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[FEATURE_COLS].astype(float)

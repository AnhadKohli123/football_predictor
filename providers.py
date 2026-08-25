"""
Fixture data providers.
=======================

Two free football APIs are supported. They return completely different JSON,
so each gets an adapter that normalises everything into the same flat dict:

    {fixture_id, league, utc_date, date, home_team, away_team, matchday,
     status, home_goals, away_goals, result}

Which one you get is decided by your token, because the two use different key
formats:

    football-data.org  32 hex characters   header: X-Auth-Token
    API-Football       40 hex characters   header: x-apisports-key

Override the guess with FOOTBALL_API_PROVIDER or --provider if needed.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

from features import result_from_goals

log = logging.getLogger("providers")

FOOTBALL_DATA = "football-data"
API_FOOTBALL = "api-football"

# Canonical short names used by the CLI, mapped to each provider's own ids.
#   key -> (display name, football-data.org id, API-Football id)
LEAGUE_TABLE: Dict[str, tuple] = {
    "PL":           ("Premier League",   2021,  39),
    "LaLiga":       ("La Liga",          2014, 140),
    "Bundesliga":   ("Bundesliga",       2002,  78),
    "SerieA":       ("Serie A",          2019, 135),
    "Ligue1":       ("Ligue 1",          2015,  61),
    "Eredivisie":   ("Eredivisie",       2003,  88),
    "PrimeiraLiga": ("Primeira Liga",    2017,  94),
    "CL":           ("Champions League", 2001,   2),
}

LEAGUE_NAMES = {key: value[0] for key, value in LEAGUE_TABLE.items()}


def detect_provider(api_key: Optional[str]) -> str:
    """Guess the provider from the token's shape."""
    if not api_key:
        return FOOTBALL_DATA
    token = api_key.strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", token):
        return API_FOOTBALL
    if re.fullmatch(r"[0-9a-fA-F]{32}", token):
        return FOOTBALL_DATA
    # RapidAPI keys are ~50 chars and not pure hex.
    if len(token) > 40:
        return API_FOOTBALL
    log.warning("could not infer the provider from the token — assuming football-data.org")
    return FOOTBALL_DATA


def current_season(today: Optional[datetime] = None) -> int:
    """
    European seasons straddle the calendar year. A match in September 2025
    belongs to season 2025; one in March 2026 also belongs to season 2025.
    """
    today = today or datetime.now(timezone.utc)
    return today.year if today.month >= 7 else today.year - 1


class BaseProvider:
    """Shared HTTP plumbing: throttling, retries, and response caching."""

    name = "base"
    base_url = ""
    min_interval = 6.5  # seconds between requests

    def __init__(self, api_key: Optional[str], cache, force: bool = False, timeout: int = 30):
        self.api_key = api_key
        self.cache = cache
        self.force = force
        self.timeout = timeout
        self.session = requests.Session()
        self.requests_made = 0
        self.failures = 0          # requests that gave up after retrying
        self._last_request_at = 0.0

    # -- to implement -------------------------------------------------

    def headers(self) -> dict:
        raise NotImplementedError

    def fetch_fixtures(self, days: int, leagues: List[str]) -> List[dict]:
        raise NotImplementedError

    def fetch_results(self, leagues: List[str], seasons: List[int]) -> List[dict]:
        raise NotImplementedError

    # -- plumbing ------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_at
        if self._last_request_at and elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def get(self, path: str, params: dict, ttl: int, max_retries: int = 3) -> Optional[dict]:
        cache_key = f"{self.name}{path}?{json.dumps(params, sort_keys=True)}"

        if not self.force:
            cached = self.cache.get_json(cache_key, ttl)
            if cached is not None:
                log.debug("cache hit: %s", cache_key)
                return cached

        if not self.api_key:
            from football_predictor import MissingAPIKey, API_KEY_HELP
            raise MissingAPIKey(API_KEY_HELP)

        url = f"{self.base_url}{path}"

        for attempt in range(1, max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(
                    url, headers=self.headers(), params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                if attempt == max_retries:
                    log.error("network error on %s: %s", path, exc)
                    self.failures += 1
                    return None
                wait = 2 ** attempt
                log.warning("network error (%s) — retrying in %ss", exc, wait)
                time.sleep(wait)
                continue
            finally:
                self._last_request_at = time.time()

            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 60))
                log.warning("rate limited — waiting %ss", wait)
                time.sleep(wait)
                continue

            if response.status_code in (401, 403):
                log.error(
                    "access denied (HTTP %s) for %s — check your token, or this "
                    "competition may not be on your plan", response.status_code, path
                )
                self.failures += 1
                return None

            if response.status_code != 200:
                if attempt == max_retries:
                    log.warning("HTTP %s for %s: %s", response.status_code, path,
                                response.text[:200])
                    self.failures += 1
                    return None
                time.sleep(2 ** attempt)
                continue

            try:
                payload = response.json()
            except ValueError:
                log.error("non-JSON response from %s", path)
                self.failures += 1
                return None

            problem = self.payload_error(payload)
            if problem:
                log.error("API error on %s: %s", path, problem)
                self.failures += 1
                return None

            self.requests_made += 1
            self.cache.put_json(cache_key, url, payload)
            return payload

        return None

    @staticmethod
    def payload_error(payload: dict) -> Optional[str]:
        """Some APIs report failure inside a 200 response."""
        self.failures += 1
        return None


# ============================================================
# football-data.org
# ============================================================

class FootballDataProvider(BaseProvider):
    name = FOOTBALL_DATA
    base_url = "https://api.football-data.org/v4"
    min_interval = 6.5  # free tier: 10 requests/minute

    def headers(self) -> dict:
        return {"X-Auth-Token": self.api_key or ""}

    def _parse(self, matches: list, display_name: str) -> List[dict]:
        rows = []
        for match in matches:
            score = match.get("score", {}).get("fullTime", {})
            home_goals, away_goals = score.get("home"), score.get("away")
            utc_date = match.get("utcDate", "")
            rows.append(
                {
                    "fixture_id": match.get("id"),
                    "league": display_name,
                    "utc_date": utc_date,
                    "date": utc_date[:10],
                    "matchday": match.get("matchday"),
                    "home_team": (match.get("homeTeam") or {}).get("name") or "Unknown",
                    "away_team": (match.get("awayTeam") or {}).get("name") or "Unknown",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result_from_goals(home_goals, away_goals),
                    "status": match.get("status"),
                    "season": (match.get("season") or {}).get("startDate", "")[:4],
                    "stage": match.get("stage", ""),
                }
            )
        return rows

    def fetch_fixtures(self, days: int, leagues: List[str]) -> List[dict]:
        today = datetime.now(timezone.utc).date()
        fixtures: List[dict] = []
        for key in leagues:
            display_name, comp_id, _ = LEAGUE_TABLE[key]
            payload = self.get(
                f"/competitions/{comp_id}/matches",
                {
                    "status": "SCHEDULED",
                    "dateFrom": today.isoformat(),
                    "dateTo": (today + timedelta(days=days)).isoformat(),
                },
                ttl=6 * 3600,
            )
            if payload:
                fixtures.extend(self._parse(payload.get("matches", []), display_name))
        return fixtures

    def fetch_results(self, leagues: List[str], seasons: List[int]) -> List[dict]:
        rows: List[dict] = []
        for key in leagues:
            display_name, comp_id, _ = LEAGUE_TABLE[key]
            for season in seasons:
                log.info("📥 %s — %s/%s", display_name, season, str(season + 1)[-2:])
                payload = self.get(
                    f"/competitions/{comp_id}/matches",
                    {"season": season, "status": "FINISHED"},
                    ttl=24 * 3600,
                )
                if payload:
                    parsed = self._parse(payload.get("matches", []), display_name)
                    log.info("   ✅ %s matches", len(parsed))
                    rows.extend(parsed)
                else:
                    log.info("   ℹ️  nothing returned")
        return rows


# ============================================================
# API-Football (api-sports.io)
# ============================================================

class ApiFootballProvider(BaseProvider):
    name = API_FOOTBALL
    base_url = "https://v3.football.api-sports.io"
    min_interval = 1.0  # free tier is a daily budget, not a tight per-minute one

    def headers(self) -> dict:
        return {"x-apisports-key": self.api_key or ""}

    @staticmethod
    def payload_error(payload: dict) -> Optional[str]:
        # This API answers 200 OK and puts failures in an "errors" field.
        errors = payload.get("errors")
        if isinstance(errors, dict) and errors:
            return "; ".join(f"{k}: {v}" for k, v in errors.items())
        if isinstance(errors, list) and errors:
            return "; ".join(str(e) for e in errors)
        return None

    def _parse(self, entries: list, display_name: str) -> List[dict]:
        rows = []
        for entry in entries:
            fixture = entry.get("fixture", {})
            teams = entry.get("teams", {})
            goals = entry.get("goals", {})
            league = entry.get("league", {})
            status = (fixture.get("status") or {}).get("short")
            utc_date = fixture.get("date", "") or ""

            # Only finished matches carry a trustworthy score.
            finished = status in {"FT", "AET", "PEN"}
            home_goals = goals.get("home") if finished else None
            away_goals = goals.get("away") if finished else None

            rows.append(
                {
                    "fixture_id": fixture.get("id"),
                    "league": display_name or league.get("name", ""),
                    "utc_date": utc_date,
                    "date": utc_date[:10],
                    "matchday": league.get("round"),
                    "home_team": (teams.get("home") or {}).get("name") or "Unknown",
                    "away_team": (teams.get("away") or {}).get("name") or "Unknown",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result_from_goals(home_goals, away_goals),
                    "status": status,
                    "season": str(league.get("season", "")),
                    "stage": league.get("round", ""),
                }
            )
        return rows

    def fetch_fixtures(self, days: int, leagues: List[str]) -> List[dict]:
        today = datetime.now(timezone.utc).date()
        season = current_season()
        fixtures: List[dict] = []
        for key in leagues:
            display_name, _, league_id = LEAGUE_TABLE[key]
            payload = self.get(
                "/fixtures",
                {
                    "league": league_id,
                    "season": season,
                    "from": today.isoformat(),
                    "to": (today + timedelta(days=days)).isoformat(),
                    "status": "NS",  # not started
                    "timezone": "UTC",
                },
                ttl=6 * 3600,
            )
            if payload:
                fixtures.extend(self._parse(payload.get("response", []), display_name))
        return fixtures

    def fetch_results(self, leagues: List[str], seasons: List[int]) -> List[dict]:
        rows: List[dict] = []
        for key in leagues:
            display_name, _, league_id = LEAGUE_TABLE[key]
            for season in seasons:
                log.info("📥 %s — %s/%s", display_name, season, str(season + 1)[-2:])
                payload = self.get(
                    "/fixtures",
                    {"league": league_id, "season": season, "status": "FT-AET-PEN",
                     "timezone": "UTC"},
                    ttl=24 * 3600,
                )
                if payload:
                    parsed = self._parse(payload.get("response", []), display_name)
                    log.info("   ✅ %s matches", len(parsed))
                    rows.extend(parsed)
                else:
                    log.info("   ℹ️  nothing returned")
        return rows


PROVIDERS = {
    FOOTBALL_DATA: FootballDataProvider,
    API_FOOTBALL: ApiFootballProvider,
}


def get_provider(api_key: Optional[str], cache, force: bool = False,
                 provider: Optional[str] = None) -> BaseProvider:
    """Build the right provider for this token."""
    name = provider or detect_provider(api_key)
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {', '.join(PROVIDERS)}")
    log.debug("using provider: %s", name)
    return PROVIDERS[name](api_key, cache, force=force)

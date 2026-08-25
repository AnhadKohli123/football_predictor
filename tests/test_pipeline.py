"""
Tests for the prediction pipeline.

    python -m unittest discover -s tests -v

These deliberately never touch the network. The provider tests feed known
payloads through the cache, which exercises exactly the same parsing path a
live response would take.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import (  # noqa: E402
    FEATURE_COLS,
    MatchHistory,
    build_features_for_match,
    result_from_goals,
    to_model_frame,
)
from football_predictor import Cache, FootballPredictor, resolve_api_key  # noqa: E402
from providers import (  # noqa: E402
    ApiFootballProvider,
    FootballDataProvider,
    current_season,
    detect_provider,
    get_provider,
)


def sample_matches() -> pd.DataFrame:
    """A small, fully hand-checkable fixture list."""
    return pd.DataFrame(
        [
            # date,        home,      away,      hg, ag
            ("2024-01-01", "Alpha",   "Beta",     3, 0),
            ("2024-01-08", "Beta",    "Alpha",    1, 1),
            ("2024-01-15", "Alpha",   "Gamma",    2, 1),
            ("2024-01-22", "Gamma",   "Alpha",    0, 4),
            ("2024-02-01", "Beta",    "Gamma",    2, 2),
            ("2024-02-08", "Alpha",   "Beta",     0, 2),
        ],
        columns=["date", "home_team", "away_team", "home_goals", "away_goals"],
    ).assign(
        date=lambda d: pd.to_datetime(d["date"]),
        league="Premier League",
        result=lambda d: [
            result_from_goals(h, a) for h, a in zip(d["home_goals"], d["away_goals"])
        ],
    )


class TestResultFromGoals(unittest.TestCase):
    def test_outcomes(self):
        self.assertEqual(result_from_goals(2, 1), "HOME_WIN")
        self.assertEqual(result_from_goals(1, 2), "AWAY_WIN")
        self.assertEqual(result_from_goals(1, 1), "DRAW")

    def test_missing_scores_give_none(self):
        self.assertIsNone(result_from_goals(None, 1))
        self.assertIsNone(result_from_goals(1, None))
        self.assertIsNone(result_from_goals(float("nan"), 1))


class TestMatchHistory(unittest.TestCase):
    def setUp(self):
        self.history = MatchHistory(sample_matches())

    def test_indexes_every_match(self):
        self.assertEqual(self.history.n_matches, 6)
        self.assertEqual(self.history.teams(), ["Alpha", "Beta", "Gamma"])

    def test_no_history_before_first_match(self):
        self.assertIsNone(self.history.team_form("Alpha", "2024-01-01"))

    def test_only_earlier_matches_count(self):
        # Before 2024-01-08 Alpha has played once: the 3-0 win on Jan 1.
        form = self.history.team_form("Alpha", "2024-01-08")
        self.assertAlmostEqual(form["form_points"], 3.0)
        self.assertAlmostEqual(form["form_goals_scored"], 3.0)
        self.assertAlmostEqual(form["form_goals_conceded"], 0.0)
        self.assertAlmostEqual(form["form_wins"], 1.0)

    def test_same_day_results_never_leak(self):
        """
        The single most important guarantee: a feature for a match on date D
        must not include D's own result, or the model trains on the answer.
        """
        frame = sample_matches()
        # Two Alpha matches on the same day, one already a thrashing.
        extra = pd.DataFrame([{
            "date": pd.Timestamp("2024-01-15"), "home_team": "Alpha",
            "away_team": "Delta", "home_goals": 9, "away_goals": 0,
            "league": "Premier League", "result": "HOME_WIN",
        }])
        history = MatchHistory(pd.concat([frame, extra], ignore_index=True))

        form = history.team_form("Alpha", "2024-01-15")
        # Only Jan 1 (3-0 W) and Jan 8 (1-1 D) may count — never the 9-0.
        self.assertAlmostEqual(form["form_goals_scored"], (3 + 1) / 2)
        self.assertLess(form["form_goals_scored"], 5.0)

    def test_form_averages_over_window(self):
        # Before 2024-02-08 Alpha has: W(3-0), D(1-1), W(2-1), W(4-0 away)
        form = self.history.team_form("Alpha", "2024-02-08")
        self.assertAlmostEqual(form["form_points"], (3 + 1 + 3 + 3) / 4)
        self.assertAlmostEqual(form["form_goals_scored"], (3 + 1 + 2 + 4) / 4)
        self.assertAlmostEqual(form["form_wins"], 3 / 4)
        self.assertAlmostEqual(form["form_draws"], 1 / 4)
        self.assertAlmostEqual(form["form_losses"], 0.0)

    def test_window_is_capped(self):
        form = self.history.team_form("Alpha", "2024-02-08", n=2)
        # Last two only: W(2-1) then W(4-0)
        self.assertAlmostEqual(form["form_points"], 3.0)

    def test_home_and_away_split(self):
        home = self.history.home_form("Alpha", "2024-02-08")
        # Alpha's home games before Feb 8: 3-0 W, 2-1 W
        self.assertAlmostEqual(home["home_win_rate"], 1.0)
        self.assertAlmostEqual(home["home_goals_scored"], 2.5)

        away = self.history.away_form("Alpha", "2024-02-08")
        # Alpha away before Feb 8: 1-1 D at Beta, 4-0 W at Gamma
        self.assertAlmostEqual(away["away_win_rate"], 0.5)
        self.assertAlmostEqual(away["away_goals_scored"], 2.5)

    def test_h2h_is_oriented_to_todays_home_team(self):
        """
        Alpha vs Beta before 2024-02-08: Alpha won 3-0 at home, then drew away.
        Asked from Alpha's side that is 1 win, 1 draw; from Beta's side it must
        be 1 loss, 1 draw — the same matches, mirrored.
        """
        from_alpha = self.history.head_to_head("Alpha", "Beta", "2024-02-08")
        self.assertAlmostEqual(from_alpha["h2h_home_wins"], 0.5)
        self.assertAlmostEqual(from_alpha["h2h_away_wins"], 0.0)
        self.assertAlmostEqual(from_alpha["h2h_draws"], 0.5)

        from_beta = self.history.head_to_head("Beta", "Alpha", "2024-02-08")
        self.assertAlmostEqual(from_beta["h2h_home_wins"], 0.0)
        self.assertAlmostEqual(from_beta["h2h_away_wins"], 0.5)
        self.assertAlmostEqual(from_beta["h2h_draws"], 0.5)

    def test_unknown_team(self):
        self.assertFalse(self.history.has_team("Nobody"))
        self.assertIsNone(self.history.team_form("Nobody", "2024-06-01"))

    def test_rows_with_missing_scores_are_dropped(self):
        frame = sample_matches()
        frame.loc[0, "home_goals"] = None
        self.assertEqual(MatchHistory(frame).n_matches, 5)


class TestFeatureVector(unittest.TestCase):
    def setUp(self):
        self.history = MatchHistory(sample_matches())

    def test_has_exactly_the_expected_keys(self):
        features = build_features_for_match(
            self.history, "Alpha", "Beta", "2024-03-01", "Premier League"
        )
        self.assertEqual(sorted(features), sorted(FEATURE_COLS))

    def test_unknown_teams_fall_back_to_priors(self):
        features = build_features_for_match(
            self.history, "Nobody", "NoOne", "2024-03-01", "Premier League"
        )
        self.assertAlmostEqual(features["home_form_points"], 1.0)
        self.assertAlmostEqual(features["h2h_draws"], 0.33)

    def test_column_order_is_stable(self):
        frame = to_model_frame(
            [
                build_features_for_match(self.history, "Alpha", "Beta", "2024-03-01", "x"),
                build_features_for_match(self.history, "Beta", "Gamma", "2024-03-01", "x"),
            ]
        )
        self.assertEqual(list(frame.columns), FEATURE_COLS)
        self.assertEqual(len(frame), 2)

    def test_league_code_for_unknown_competition(self):
        features = build_features_for_match(
            self.history, "Alpha", "Beta", "2024-03-01", "Not A Real League"
        )
        self.assertEqual(features["league_code"], -1)

    def test_neutral_flag_for_world_cup(self):
        wc = build_features_for_match(self.history, "Alpha", "Beta", "2024-03-01", "World Cup")
        pl = build_features_for_match(self.history, "Alpha", "Beta", "2024-03-01", "Premier League")
        self.assertEqual(wc["is_neutral"], 1)
        self.assertEqual(pl["is_neutral"], 0)


class TestProviderDetection(unittest.TestCase):
    def test_key_formats(self):
        self.assertEqual(detect_provider("a" * 40), "api-football")
        self.assertEqual(detect_provider("b" * 32), "football-data")

    def test_long_non_hex_key_is_rapidapi_style(self):
        self.assertEqual(detect_provider("x" * 50), "api-football")

    def test_season_straddles_the_calendar_year(self):
        self.assertEqual(current_season(datetime(2025, 9, 1, tzinfo=timezone.utc)), 2025)
        self.assertEqual(current_season(datetime(2026, 3, 1, tzinfo=timezone.utc)), 2025)
        self.assertEqual(current_season(datetime(2026, 7, 1, tzinfo=timezone.utc)), 2026)


class ProviderTestCase(unittest.TestCase):
    """Serves canned payloads through the cache so no network call happens."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Cache(str(Path(self.tmp.name) / "cache.db"))

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def prime(self, provider, path, params, payload):
        key = f"{provider.name}{path}?{json.dumps(params, sort_keys=True)}"
        self.cache.put_json(key, path, payload)


class TestFootballDataProvider(ProviderTestCase):
    def test_parses_fixtures(self):
        provider = FootballDataProvider("b" * 32, self.cache)
        today = datetime.now(timezone.utc).date()
        params = {
            "status": "SCHEDULED",
            "dateFrom": today.isoformat(),
            "dateTo": (today + timedelta(days=7)).isoformat(),
        }
        self.prime(
            provider, "/competitions/2021/matches", params,
            {
                "matches": [
                    {
                        "id": 1001,
                        "utcDate": "2026-09-01T14:00:00Z",
                        "matchday": 4,
                        "status": "SCHEDULED",
                        "homeTeam": {"name": "Arsenal FC"},
                        "awayTeam": {"name": "Chelsea FC"},
                        "score": {"fullTime": {"home": None, "away": None}},
                        "season": {"startDate": "2026-08-10"},
                    }
                ]
            },
        )
        fixtures = provider.fetch_fixtures(days=7, leagues=["PL"])
        self.assertEqual(len(fixtures), 1)
        fixture = fixtures[0]
        self.assertEqual(fixture["fixture_id"], 1001)
        self.assertEqual(fixture["home_team"], "Arsenal FC")
        self.assertEqual(fixture["away_team"], "Chelsea FC")
        self.assertEqual(fixture["date"], "2026-09-01")
        self.assertEqual(fixture["league"], "Premier League")
        self.assertIsNone(fixture["result"])

    def test_parses_finished_results(self):
        provider = FootballDataProvider("b" * 32, self.cache)
        self.prime(
            provider, "/competitions/2021/matches",
            {"season": 2024, "status": "FINISHED"},
            {
                "matches": [
                    {
                        "id": 2002,
                        "utcDate": "2024-09-01T14:00:00Z",
                        "matchday": 4,
                        "status": "FINISHED",
                        "homeTeam": {"name": "Arsenal FC"},
                        "awayTeam": {"name": "Chelsea FC"},
                        "score": {"fullTime": {"home": 2, "away": 1}},
                        "season": {"startDate": "2024-08-10"},
                    }
                ]
            },
        )
        rows = provider.fetch_results(leagues=["PL"], seasons=[2024])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["result"], "HOME_WIN")
        self.assertEqual(rows[0]["home_goals"], 2)

    def test_missing_team_name_does_not_crash(self):
        provider = FootballDataProvider("b" * 32, self.cache)
        self.prime(
            provider, "/competitions/2021/matches",
            {"season": 2024, "status": "FINISHED"},
            {"matches": [{"id": 3, "utcDate": "2024-09-01T14:00:00Z",
                          "homeTeam": {}, "awayTeam": None,
                          "score": {"fullTime": {"home": 1, "away": 0}}}]},
        )
        rows = provider.fetch_results(leagues=["PL"], seasons=[2024])
        self.assertEqual(rows[0]["home_team"], "Unknown")
        self.assertEqual(rows[0]["away_team"], "Unknown")


class TestApiFootballProvider(ProviderTestCase):
    def api_football_fixture(self, status="NS", home_goals=None, away_goals=None):
        return {
            "fixture": {
                "id": 555,
                "date": "2026-09-01T14:00:00+00:00",
                "status": {"short": status},
            },
            "league": {"id": 39, "name": "Premier League", "season": 2026, "round": "Regular Season - 4"},
            "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Chelsea"}},
            "goals": {"home": home_goals, "away": away_goals},
        }

    def test_parses_fixtures(self):
        provider = ApiFootballProvider("a" * 40, self.cache)
        today = datetime.now(timezone.utc).date()
        params = {
            "league": 39, "season": current_season(),
            "from": today.isoformat(), "to": (today + timedelta(days=7)).isoformat(),
            "status": "NS", "timezone": "UTC",
        }
        self.prime(provider, "/fixtures", params,
                   {"errors": [], "response": [self.api_football_fixture()]})

        fixtures = provider.fetch_fixtures(days=7, leagues=["PL"])
        self.assertEqual(len(fixtures), 1)
        self.assertEqual(fixtures[0]["fixture_id"], 555)
        self.assertEqual(fixtures[0]["home_team"], "Arsenal")
        self.assertEqual(fixtures[0]["date"], "2026-09-01")
        self.assertEqual(fixtures[0]["league"], "Premier League")

    def test_scores_only_trusted_when_finished(self):
        provider = ApiFootballProvider("a" * 40, self.cache)
        params = {"league": 39, "season": 2024, "status": "FT-AET-PEN", "timezone": "UTC"}

        # An in-progress match must not contribute a result.
        self.prime(provider, "/fixtures", params,
                   {"response": [self.api_football_fixture("1H", 1, 0)]})
        rows = provider.fetch_results(leagues=["PL"], seasons=[2024])
        self.assertIsNone(rows[0]["result"])
        self.assertIsNone(rows[0]["home_goals"])

    def test_finished_match_yields_result(self):
        provider = ApiFootballProvider("a" * 40, self.cache)
        params = {"league": 39, "season": 2024, "status": "FT-AET-PEN", "timezone": "UTC"}
        self.prime(provider, "/fixtures", params,
                   {"response": [self.api_football_fixture("FT", 2, 1)]})
        rows = provider.fetch_results(leagues=["PL"], seasons=[2024])
        self.assertEqual(rows[0]["result"], "HOME_WIN")
        self.assertEqual(rows[0]["home_goals"], 2)

    def test_errors_inside_a_200_response_are_detected(self):
        """This API answers HTTP 200 and reports failure in an 'errors' field."""
        self.assertIsNotNone(
            ApiFootballProvider.payload_error({"errors": {"token": "invalid"}})
        )
        self.assertIsNotNone(
            ApiFootballProvider.payload_error({"errors": ["quota reached"]})
        )
        self.assertIsNone(ApiFootballProvider.payload_error({"errors": []}))
        self.assertIsNone(ApiFootballProvider.payload_error({"errors": {}}))
        self.assertIsNone(ApiFootballProvider.payload_error({"response": []}))


class TestCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Cache(str(Path(self.tmp.name) / "c.db"))

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def test_roundtrip(self):
        self.cache.put_json("k", "http://x", {"a": 1})
        self.assertEqual(self.cache.get_json("k", ttl=3600), {"a": 1})

    def test_expired_entry_is_ignored(self):
        self.cache.put_json("k", "http://x", {"a": 1})
        self.assertIsNone(self.cache.get_json("k", ttl=-1))

    def test_missing_key(self):
        self.assertIsNone(self.cache.get_json("nope", ttl=3600))

    def test_fixtures_without_ids_are_skipped(self):
        self.cache.save_fixtures([{"fixture_id": None, "league": "L", "utc_date": "d",
                                   "home_team": "h", "away_team": "a"}])
        count = self.cache.conn.execute("SELECT COUNT(*) FROM upcoming_fixtures").fetchone()[0]
        self.assertEqual(count, 0)


class TestEndToEnd(unittest.TestCase):
    """Trains a real model on synthetic data, then predicts through the CLI path."""

    @classmethod
    def setUpClass(cls):
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "models").mkdir()
        (root / "data").mkdir()

        # A synthetic league with a genuine home advantage.
        rng = np.random.default_rng(0)
        teams = [f"Team {i}" for i in range(8)]
        rows = []
        start = pd.Timestamp("2023-01-01")
        for week in range(120):
            for _ in range(4):
                home, away = rng.choice(teams, 2, replace=False)
                hg, ag = rng.poisson(1.6), rng.poisson(1.1)
                rows.append(
                    {
                        "league": "Premier League",
                        "date": start + pd.Timedelta(days=7 * week),
                        "home_team": home, "away_team": away,
                        "home_goals": hg, "away_goals": ag,
                        "result": result_from_goals(hg, ag),
                    }
                )
        matches = pd.DataFrame(rows)
        cls.history_path = root / "data" / "all_matches.csv"
        matches.to_csv(cls.history_path, index=False)

        history = MatchHistory(matches)
        features = to_model_frame(
            [
                build_features_for_match(history, r.home_team, r.away_team, r.date, r.league)
                for r in matches.itertuples(index=False)
            ]
        )
        encoder = LabelEncoder()
        y = encoder.fit_transform(matches["result"])
        model = RandomForestClassifier(n_estimators=25, random_state=0).fit(features, y)

        cls.model_path = root / "models" / "predictor.pkl"
        cls.encoder_path = root / "models" / "label_encoder.pkl"
        joblib.dump(model, cls.model_path)
        joblib.dump(encoder, cls.encoder_path)
        cls.db_path = str(root / "cache.db")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make_predictor(self):
        return FootballPredictor(
            model_path=str(self.model_path),
            encoder_path=str(self.encoder_path),
            history_path=str(self.history_path),
            db_path=self.db_path,
            api_key="a" * 40,
        )

    def test_single_match_prediction(self):
        with self.make_predictor() as predictor:
            prediction = predictor.predict_match("Team 1", "Team 2", "2025-06-01",
                                                 "Premier League")
        self.assertIn(prediction.prediction, {"Home Win", "Draw", "Away Win"})
        total = prediction.prob_home_win + prediction.prob_draw + prediction.prob_away_win
        self.assertAlmostEqual(total, 1.0, places=5)
        self.assertIn(prediction.confidence, {"Low", "Medium", "High"})

    def test_batch_matches_single_prediction(self):
        """Batched and one-at-a-time must agree, or the CSV lies about the table."""
        fixtures = [
            {"fixture_id": 1, "league": "Premier League", "date": "2025-06-01",
             "utc_date": "2025-06-01T12:00:00Z", "home_team": "Team 1", "away_team": "Team 2"},
            {"fixture_id": 2, "league": "Premier League", "date": "2025-06-02",
             "utc_date": "2025-06-02T12:00:00Z", "home_team": "Team 3", "away_team": "Team 4"},
        ]
        with self.make_predictor() as predictor:
            batch = predictor.predict_fixtures(fixtures)
            singles = [
                predictor.predict_match(f["home_team"], f["away_team"], f["date"], f["league"])
                for f in fixtures
            ]
        self.assertEqual(len(batch), 2)
        for b, s in zip(batch, singles):
            self.assertEqual(b.prediction, s.prediction)
            self.assertAlmostEqual(b.prob_home_win, s.prob_home_win, places=6)

    def test_export_writes_both_formats(self):
        fixtures = [
            {"fixture_id": 7, "league": "Premier League", "date": "2025-06-01",
             "utc_date": "2025-06-01T12:00:00Z", "home_team": "Team 1", "away_team": "Team 2"}
        ]
        out = Path(self.tmp.name)
        with self.make_predictor() as predictor:
            predictions = predictor.predict_fixtures(fixtures)
            predictor.export(predictions, str(out / "p.csv"), str(out / "p.json"))

        csv = pd.read_csv(out / "p.csv")
        self.assertEqual(len(csv), 1)
        self.assertIn("prob_home_win", csv.columns)
        payload = json.loads((out / "p.json").read_text())
        self.assertEqual(payload[0]["home_team"], "Team 1")

    def test_unknown_teams_are_reported(self):
        fixtures = [
            {"fixture_id": 9, "league": "Premier League", "date": "2025-06-01",
             "utc_date": "2025-06-01T12:00:00Z",
             "home_team": "Team 1", "away_team": "Never Heard Of Them"}
        ]
        with self.make_predictor() as predictor:
            self.assertEqual(predictor.unknown_teams(fixtures), ["Never Heard Of Them"])

    def test_fixtures_served_from_cache_make_no_request(self):
        with self.make_predictor() as predictor:
            provider = predictor.client
            today = datetime.now(timezone.utc).date()
            params = {
                "league": 39, "season": current_season(),
                "from": today.isoformat(), "to": (today + timedelta(days=5)).isoformat(),
                "status": "NS", "timezone": "UTC",
            }
            key = f"{provider.name}/fixtures?{json.dumps(params, sort_keys=True)}"
            predictor.cache.put_json(key, "/fixtures", {
                "response": [{
                    "fixture": {"id": 42, "date": "2026-09-01T14:00:00+00:00",
                                "status": {"short": "NS"}},
                    "league": {"id": 39, "name": "Premier League", "season": current_season(),
                               "round": "Regular Season - 4"},
                    "teams": {"home": {"name": "Team 1"}, "away": {"name": "Team 2"}},
                    "goals": {"home": None, "away": None},
                }]
            })
            fixtures = predictor.fetch_upcoming_fixtures(days=5, leagues=["PL"])
            predictions = predictor.predict_fixtures(fixtures)

        self.assertEqual(len(fixtures), 1)
        self.assertEqual(provider.requests_made, 0)
        self.assertEqual(provider.failures, 0)
        self.assertEqual(predictions[0].home_team, "Team 1")


class TestKeyResolution(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in
                      ("FOOTBALL_API_KEY", "FOOTBALL_DATA_API_TOKEN", "API_FOOTBALL_KEY")}
        for key in self.saved:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_explicit_wins(self):
        os.environ["FOOTBALL_API_KEY"] = "from_env"
        self.assertEqual(resolve_api_key("explicit"), "explicit")

    def test_environment_is_used(self):
        os.environ["FOOTBALL_API_KEY"] = "from_env"
        self.assertEqual(resolve_api_key(), "from_env")

    def test_placeholder_is_rejected(self):
        os.environ["FOOTBALL_API_KEY"] = "YOUR_TOKEN_HERE"
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as empty:
            os.chdir(empty)   # so no real .env is picked up
            try:
                self.assertIsNone(resolve_api_key())
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main(verbosity=2)

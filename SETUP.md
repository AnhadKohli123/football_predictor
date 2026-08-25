# Architecture & integration guide

How the pieces fit together, and where to change things.

## The pipeline

```
                      ┌────────────────────┐
  openfootball ──────▶│ fetch_club_data.py │  ← no token needed
  API ───────────────▶│ data_fetcher.py    │──▶ data/all_matches.csv
  Kaggle CSV ────────▶│ merge_wc.py        │        (raw results)
                      └────────────────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────┐
                                          │ build_features.py  │
                                          │   uses features.py │──▶ data/features.csv
                                          └────────────────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────┐
                                          │  train_model.py    │──▶ models/predictor.pkl
                                          └────────────────────┘
                                                       │
                                                       ▼
  predict.py ─┐
  app.py ─────┴▶ FootballPredictor ──▶ providers.py ──▶ API (cached in SQLite)
                        │
                        └──▶ features.py ──▶ model ──▶ predictions.csv / .json
```

## Modules

| File | Responsibility |
|---|---|
| `features.py` | Feature definitions and the indexed match history. The single source of truth. |
| `providers.py` | API adapters. Normalises two very different JSON shapes into one dict. |
| `football_predictor.py` | The engine: SQLite cache, lazy model loading, prediction. |
| `predict.py` | Argument parsing and output formatting. No business logic. |
| `app.py` | Flask frontend. Thin JSON wrapper over the same engine. |
| `fetch_club_data.py` | Club results from openfootball. Needs no API token. |

## Why features.py is shared

Training and inference must build features identically. The original version
of this project had the feature code copied into both `build_features.py` and
`predict.py`. That works right up until someone edits one copy — after which
the model is silently fed vectors whose columns mean something different from
what it was trained on. The predictions still look plausible, which is what
makes it dangerous.

Everything now imports from `features.py`, and `FEATURE_COLS` fixes the column
order. `FootballPredictor._load_model` additionally compares the model's
recorded `feature_names_in_` against `FEATURE_COLS` and warns on a mismatch.

## Team-name matching

Every source spells clubs differently:

| Source | Spelling |
|---|---|
| openfootball | `Arsenal FC`, `Bayern München` |
| football-data.org | `Arsenal FC`, `FC Bayern München` |
| API-Football | `Arsenal`, `Bayern Munich` |
| A person typing | `arsenal`, `Man City`, `Spurs` |

`features.team_key()` normalises all of these to a common key: accents are
stripped, punctuation removed, club-type words (`FC`, `AFC`, `CF`, `SC`…)
dropped, and a small alias table handles what is left (`Spurs` →
`tottenham hotspur`). `MatchHistory` indexes on that key and keeps the
original spelling for display.

This matters more than it looks. Without it, every fixture from the API would
look like a team the model had never seen, silently fall back to
league-average priors, and produce confident-looking output carrying no
information at all. `test_distinct_clubs_stay_distinct` guards the opposite
failure — Manchester City and Manchester United must never collapse together.

When you add an alias, add a test alongside it.

## The as-of rule

A feature for a match on date *D* may only use matches that finished strictly
**before** *D*. Same-day matches are excluded too, since you would not know
their results when predicting a kickoff earlier that day.

This is enforced in one place — `MatchHistory._window` uses `bisect_left`
against the match date, so every lookup gets the same guarantee. There is a
test (`test_same_day_results_never_leak`) that pins the behaviour down.

Get this wrong and your reported accuracy climbs while real accuracy does not.

## Performance

`MatchHistory` buckets matches by team and by opponent pair once, keeps each
bucket sorted, and binary-searches it. The naive alternative — re-filtering
the whole DataFrame per lookup — is O(n²) overall.

Measured on the 23,551-match dataset: **9.7 minutes → 2.4 seconds** (~240×).
The gap widens as the dataset grows.

## Adding a feature

1. Add the key to `FEATURE_COLS` in `features.py`
2. Compute it in `build_features_for_match`, taking values only from
   `MatchHistory` (which enforces the as-of rule)
3. Add a neutral fallback for teams with no history
4. Re-run `build_features.py` then `train_model.py` — both, in that order

Inference picks the new feature up automatically. Skipping the retrain gets
you the mismatch warning rather than silently wrong output.

Ideas worth trying, roughly in order of expected value:

- **Rest days** since each side's previous match (fixture congestion is real)
- **League table position** and points gap
- **Longer form windows** alongside the current 5 — form at 5 and 10 games
  carries different information
- **Elo rating** per team, updated match by match. Usually the single
  strongest addition to a form-based model
- **Market odds**, if you can get them. They encode information no public
  dataset has, and will beat everything else here

Club football is a harder problem than international football (49% vs 56% for
the same model), because league opponents are far more evenly matched. Do not
read a lower number here as a worse model.

## Adding a provider

Subclass `BaseProvider` in `providers.py` and implement:

```python
class MyProvider(BaseProvider):
    name = "my-provider"
    base_url = "https://api.example.com"
    min_interval = 1.0                      # seconds between requests

    def headers(self) -> dict: ...
    def fetch_fixtures(self, days, leagues) -> list[dict]: ...
    def fetch_results(self, leagues, seasons) -> list[dict]: ...

    @staticmethod
    def payload_error(payload):             # if the API reports errors in a 200
        ...
```

Return dicts with these keys: `fixture_id`, `league`, `utc_date`, `date`,
`home_team`, `away_team`, `matchday`, `status`, `home_goals`, `away_goals`,
`result`, `season`, `stage`. Add the competition ids to `LEAGUE_TABLE` and
register the class in `PROVIDERS`.

Retries, throttling, caching and failure counting are all handled by the base
class — do not reimplement them.

## Caching

SQLite, in `football_cache.db`. Three tables:

| Table | Holds | Freshness |
|---|---|---|
| `api_cache` | Raw JSON responses, keyed by provider + path + params | 6h fixtures, 24h results |
| `upcoming_fixtures` | Flattened fixture list | overwritten per fetch |
| `predictions` | Every prediction made, with its probabilities | never expires |

`--force` bypasses reads but still writes. Deleting the file is always safe;
it will be rebuilt on the next run.

The `predictions` table exists so you can score yourself later: join it
against actual results and fill in the `actual` column to measure how the
model does on fixtures it had never seen at prediction time. That number is
worth far more than the training metrics.

## API quota

Fixtures are cached for six hours, so a normal prediction run costs one
request per league, and re-running within the same afternoon costs nothing.

A full historical backfill is the expensive part: one request per league per
season, about 40 requests for five seasons across eight competitions. Both
free tiers cover that comfortably if you are not repeating it daily.

- **API-Football** free: 100 requests/day
- **football-data.org** free: 10 requests/minute

Better still, build the training set with `fetch_club_data.py`, which pulls
from a public GitHub dataset and spends no quota at all. Then your token is
needed only for upcoming fixtures — a handful of requests per run.

## Web frontend

`app.py` is a thin Flask layer over `FootballPredictor`; all the logic lives
in the engine, so the CLI and the browser cannot disagree.

| Endpoint | Returns |
|---|---|
| `GET /api/status` | Model, history and token state — drives the header |
| `GET /api/teams` | Known teams, for the pickers |
| `POST /api/predict` | One fixture. Works offline |
| `GET /api/fixtures` | Upcoming fixtures, predicted. Needs a token |
| `GET /api/history` | Previously logged predictions |

A prediction resting on priors is flagged `reliable: false`, and the UI shows
a warning rather than presenting it as equivalent to a well-supported one.

Bind to localhost only unless you mean otherwise — `--host 0.0.0.0` exposes
the app, and its Flask dev server is not hardened for that.

`data_fetcher.py` merges into the existing CSV rather than replacing it, so a
run interrupted halfway can simply be repeated without re-spending quota on
what it already got.

## Testing

```bash
python -m unittest discover -s tests -v
```

Provider tests prime the SQLite cache with a known payload and read it back
through the normal code path, so parsing is genuinely exercised without any
network access. Add a test there when you add a provider.

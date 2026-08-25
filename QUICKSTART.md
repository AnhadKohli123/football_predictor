# Quickstart

Seven steps from a clean clone to predictions.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Add your API token

Create a file called `.env` next to `predict.py`:

```
FOOTBALL_API_KEY=your_token_here
```

`.env` is gitignored, so it will not be committed. Free tokens:

| Provider | Sign up | Key format |
|---|---|---|
| API-Football | <https://dashboard.api-football.com/register> | 40 hex characters |
| football-data.org | <https://www.football-data.org/client/register> | 32 hex characters |

The provider is detected from the key's format. Override it with
`--provider api-football` or `--provider football-data` if the guess is wrong.

## 3. Check the token works

Spend one request before spending your whole quota:

```bash
python data_fetcher.py --league PL --seasons 2024 -v
```

If that prints match counts, you are good. If it prints `access denied`, the
token or the provider is wrong.

## 4. Download history

```bash
python data_fetcher.py
```

Writes `data/all_matches.csv`. Re-running merges rather than overwrites, so an
interrupted run is safe to repeat.

**No token yet?** You can still build a working model offline from the Kaggle
international-results file already in `data/`:

```bash
python merge_wc.py --international-only \
  --tournaments "FIFA World Cup" "UEFA Euro" "Copa América" --since 1994
```

## 5. Build features

```bash
python build_features.py
```

Writes `data/features.csv` — 29 features per match.

## 6. Train

```bash
python train_model.py
```

Writes `models/predictor.pkl`. Read the printed **log loss vs. class priors**
line: if the model is not beating the priors, more data will help more than
more tuning will.

## 7. Predict

```bash
python predict.py                              # next 14 days, all leagues
python predict.py --days 7 --league PL         # one week of Premier League
python predict.py --match "Arsenal" "Chelsea"  # a single fixture, offline
```

Results print as a table and save to `predictions.csv` / `predictions.json`.

---

## Everyday use

Once set up, this is the only command you need:

```bash
python predict.py --days 7
```

Fixtures are cached for six hours, so re-running the same afternoon costs no
API quota at all.

## Refresh occasionally

The model only knows matches up to the last fetch. Every week or two:

```bash
python data_fetcher.py --seasons 2026   # top up recent results
python build_features.py
python train_model.py
```

## If something breaks

| Symptom | Fix |
|---|---|
| `No football API token found` | Create `.env` (step 2) |
| `request(s) failed` | Wrong provider — try `--provider football-data` |
| `No trained model` | Run steps 4–6 |
| `No scheduled fixtures found` | Off-season; try `--days 30` |
| `No history for: <teams>` | Run `data_fetcher.py` for that league |

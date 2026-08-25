# Quickstart

From a clean clone to predictions. **Steps 1–4 need no API token at all** —
the training data comes from a public GitHub dataset. A token is only needed
for step 6, upcoming fixtures.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Download club results (no token needed)

```bash
python fetch_club_data.py --all-leagues --seasons 14
```

Roughly 44,000 matches across the big five leagues, the Eredivisie, the
Primeira Liga and three second tiers, from 2012 to today. Writes
`data/all_matches.csv`.

Smaller and faster, if you prefer:

```bash
python fetch_club_data.py --seasons 6          # big five only
python fetch_club_data.py --league PL --league LaLiga
```

Including the second tiers is worth it: promoted clubs then arrive in the top
flight with a real record instead of falling back to league-average priors.

## 3. Build features

```bash
python build_features.py
```

Writes `data/features.csv` — 29 features per match. Takes a few seconds.

## 4. Train

```bash
python train_model.py
```

Writes `models/predictor.pkl`. Read the printed **log loss vs. class priors**
line: if the model is not beating the priors, it has learned nothing useful
and more data will help more than more tuning will.

## 5. Predict

In the browser:

```bash
python app.py          # then open http://127.0.0.1:5000
```

Or in the terminal:

```bash
python predict.py --match "Arsenal" "Chelsea"
```

Both work entirely offline. You now have a working predictor.

## 6. (Optional) Add your API token

Only needed to fetch *upcoming* fixtures. Create a file called `.env` next to
`predict.py`:

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

## 7. Fetch upcoming fixtures

Spend one request first, to check the token and provider are right:

```bash
python predict.py --days 7 --league PL -v
```

Then use it normally:

```bash
python predict.py                          # next 14 days, all leagues
python predict.py --days 7 --league PL     # one week of Premier League
python predict.py --force                  # ignore the cache, refetch
```

Results print as a table and save to `predictions.csv` / `predictions.json`.
The **Upcoming Fixtures** tab in `python app.py` does the same in the browser.

---

## Everyday use

```bash
python app.py                 # browser
python predict.py --days 7    # terminal
```

Fixtures are cached for six hours, so re-running the same afternoon costs no
API quota at all.

## Refresh occasionally

The model only knows matches up to the last fetch. Every week or two:

```bash
python fetch_club_data.py --all-leagues --seasons 14
python build_features.py
python train_model.py
```

`fetch_club_data.py` needs no token, so this costs nothing and can be run as
often as you like.

## If something breaks

| Symptom | Fix |
|---|---|
| `No football API token found` | Create `.env` (step 6) — only fixtures need it |
| `request(s) failed` | Wrong provider — try `--provider football-data` |
| `No trained model` | Run steps 2–4 |
| `No scheduled fixtures found` | Off-season; try `--days 30` |
| `No history for: <teams>` | Newly promoted club — refetch with `--all-leagues` |
| Web page loads but cannot predict | Run steps 2–4, then restart `app.py` |

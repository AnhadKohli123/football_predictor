# ⚽ Football Match Predictor

Predicts Home Win / Draw / Away Win for club matches in Europe's top leagues,
using a gradient-boosted model trained on 44,000+ historical results.

Runs **on demand**. There is no scheduler and no background job — you run it,
it fetches only the fixtures in the window you asked for, and reuses anything
it already has that is still fresh.

```bash
python predict.py --days 7 --league PL
```

```
DATE        HOME       AWAY       PREDICTION  HOME%  DRAW%  AWAY%  CONF

  ── Premier League ──
2026-09-01  Arsenal    Chelsea    Home Win    54.2%  24.1%  21.7%  ✅ Medium
2026-09-02  Liverpool  Everton    Home Win    67.4%  19.5%  13.1%  🔥 High
```

## What it does

- Fetches upcoming fixtures from **API-Football** or **football-data.org** —
  whichever your API key belongs to, detected automatically
- Builds 29 features per match: recent form, home/away splits, head-to-head,
  and the differences between the two sides
- Predicts with XGBoost or Random Forest, whichever scored better in training
- Caches everything in SQLite, so repeat runs cost no API quota
- Exports to CSV and JSON, and logs every prediction for later scoring

## Install

```bash
git clone https://github.com/AnhadKohli123/football_predictor.git
cd football_predictor
./setup.sh
```

That is the whole install. `setup.sh` creates a virtualenv, installs
dependencies, downloads 44,000 club matches, builds features and trains the
model — about a minute, and **no API token is needed for any of it**. Re-run
it any time; it skips work that is already done unless you pass `--force`.

Then:

```bash
source .venv/bin/activate
python app.py            # http://127.0.0.1:5000
```

<details>
<summary>Manual install, or Windows</summary>

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python fetch_club_data.py --all-leagues --seasons 14
python build_features.py
python train_model.py
python app.py
```

A virtualenv is not optional on Debian, Ubuntu or Homebrew Python: those mark
the system interpreter as externally managed (PEP 668) and a plain
`pip install` fails with `externally-managed-environment`. If `python -m venv`
itself fails on Debian or Ubuntu, run `sudo apt install python3-venv` first.

</details>

### Upcoming fixtures need a token

Predictions work offline. Only the fixtures tab and `predict.py --days N`
call an API. `.env` is gitignored and is never in the repo, so create it on
each machine:

```bash
echo 'FOOTBALL_API_KEY=your_token_here' > .env
```

Free tokens: [API-Football](https://dashboard.api-football.com/register)
(40-character key) or
[football-data.org](https://www.football-data.org/client/register)
(32-character key). The provider is detected from the key's format; override
with `--provider` if the guess is wrong. Restart `app.py` after editing `.env`.

**[QUICKSTART.md](QUICKSTART.md)** walks through it step by step;
**[SETUP.md](SETUP.md)** explains the architecture.

## Web interface

```bash
python app.py
```

A local dashboard at `http://127.0.0.1:5000` with three tabs: a match
predictor with searchable team pickers, an upcoming-fixtures view, and the
model's own performance figures. Team names are matched loosely, so "Arsenal",
"Arsenal FC" and "arsenal" all resolve to the same club.

## Usage

```bash
python predict.py                              # next 14 days, every league
python predict.py --days 7 --league PL         # one league, one week
python predict.py --league PL --league CL      # several leagues
python predict.py --match "Arsenal" "Chelsea"  # one fixture, no network
python predict.py --min-confidence High        # only the strong signals
python predict.py --force                      # ignore the cache, refetch
```

Leagues: `PL`, `LaLiga`, `SerieA`, `Bundesliga`, `Ligue1`, `Eredivisie`,
`PrimeiraLiga`, `CL`.

## How well does it work?

On a chronological hold-out of 8,874 club matches (everything after
2023-11-27, never seen during training):

| Metric | Model | Baseline |
|---|---|---|
| Accuracy | **49.0%** | 43.3% (always predict home win) |
| Log loss | **1.026** | 1.075 (training-set class priors) |

It beats both baselines, which is the bar that matters — but note the margin
is modest, and smaller than the same model achieves on international football
(56.1%). That is not a worse model; club leagues are simply harder. A league
table is built to be competitive, whereas a World Cup group stage regularly
throws up genuine mismatches that are easy to call.

Three things worth understanding before you trust a number like that:

**The split is chronological, not random.** The model trains on older matches
and is tested on more recent ones. A random split would let it train on 2025
matches and be tested on 2019 ones — it would already know how those teams
turned out, and the accuracy would look far better than anything you would
ever see on real fixtures.

**Draws are the hard part.** Home and away wins are predicted reasonably well;
draws are barely predicted at all. That is not a bug in this model so much as
a property of the problem — draws rarely have the strongest signal, so a
probabilistic model almost never ranks one first. If you care about draws,
read the `DRAW%` column rather than the headline prediction.

**Team names are matched loosely.** Sources spell clubs differently —
openfootball says "Arsenal FC", API-Football says "Arsenal". Names are matched
on a normalised key so these resolve to the same club. Without that, every
fixture would look like an unfamiliar team and quietly fall back to priors:
the predictions would still render, they would just be meaningless.

Roughly 48–53% is the honest range for club football from form data alone.
Treat the output as odds, not answers.

## Project layout

```
football_predictor/
├── app.py                  # web frontend (Flask)
├── predict.py              # CLI entry point
├── football_predictor.py   # engine: caching, fixtures, prediction
├── providers.py            # API adapters (API-Football, football-data.org)
├── features.py             # feature definitions + team-name matching
├── fetch_club_data.py      # step 1: club results, no token needed
├── data_fetcher.py         # step 1 (alt): results via your API token
├── build_features.py       # step 2: results → features
├── train_model.py          # step 3: train and evaluate
├── merge_wc.py             # fold international results in as well
├── templates/, static/     # frontend
├── tests/test_pipeline.py  # 53 tests, no network required
├── requirements.txt
├── .env                    # your API token (gitignored)
├── data/                   # datasets (generated files are gitignored)
└── models/                 # trained model (gitignored)
```

`features.py` is deliberately shared between training and inference. If the
two ever computed features differently, the model would be fed vectors that
mean something different from what it learned, and the predictions would be
wrong in a way that is very hard to notice.

## Tests

```bash
python -m unittest discover -s tests -v
```

53 tests covering feature correctness, team-name matching, score parsing,
both API adapters, caching, and an end-to-end train-and-predict cycle. None of
them touch the network.

## Security note

Never commit an API token. Keys belong in `.env`, which is gitignored. If a
key has ever been pasted into a commit, a chat, or a screenshot, rotate it —
tokens in a public repository's history are scraped within minutes.

## Limitations

- Form-based features only: no injuries, suspensions, lineups, xG or odds
- No model of fixture congestion, travel distance or motivation
- Draws are rarely the top prediction — read the `DRAW%` column for those
- Newly promoted clubs have thin history; fetch second tiers with
  `--all-leagues` so they arrive with a real record instead of priors
- Name matching handles common variants, but an unusual spelling can still
  miss; the CLI and the web UI both flag teams with no history

## License

MIT — see [LICENSE](LICENSE).

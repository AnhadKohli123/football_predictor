# ⚽ Football Match Predictor

Predicts Home Win / Draw / Away Win for upcoming matches in Europe's top
leagues and the Champions League, using a gradient-boosted model trained on
historical results.

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
pip install -r requirements.txt
```

Add your token to a `.env` file (gitignored):

```
FOOTBALL_API_KEY=your_token_here
```

Then follow **[QUICKSTART.md](QUICKSTART.md)** — seven steps to your first
prediction. **[SETUP.md](SETUP.md)** explains the architecture.

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

On a chronological hold-out of 4,711 international matches:

| Metric | Model | Baseline |
|---|---|---|
| Accuracy | **56.1%** | 47.8% (always predict home win) |
| Log loss | **0.939** | 1.052 (training-set class priors) |

Two things worth understanding before you trust a number like that:

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

Roughly 50–56% is the honest range for football result prediction from form
data alone. Treat the output as odds, not answers.

## Project layout

```
football_predictor/
├── predict.py              # CLI entry point
├── football_predictor.py   # engine: caching, fixtures, prediction
├── providers.py            # API adapters (API-Football, football-data.org)
├── features.py             # feature definitions — shared by train and predict
├── data_fetcher.py         # step 1: download historical results
├── build_features.py       # step 2: results → features
├── train_model.py          # step 3: train and evaluate
├── merge_wc.py             # fold international results into the dataset
├── tests/test_pipeline.py  # 39 tests, no network required
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

39 tests covering feature correctness, both API adapters, caching, and an
end-to-end train-and-predict cycle. None of them touch the network.

## Security note

Never commit an API token. Keys belong in `.env`, which is gitignored. If a
key has ever been pasted into a commit, a chat, or a screenshot, rotate it —
tokens in a public repository's history are scraped within minutes.

## Limitations

- Form-based features only: no injuries, suspensions, lineups, xG or odds
- No model of fixture congestion, travel distance or motivation
- Cup competitions with two-legged ties are treated as independent matches
- Team names must match between the API and your history file; a rename or a
  different spelling means the team looks brand new and falls back to priors

## License

MIT — see [LICENSE](LICENSE).

# football_predictor
A model for predicting the outcomes of football games

#repo structure
├── data/               ← match CSVs stored here
├── football_data_fetcher.py   ← the script we just built
├── build_features.py          ← next script (Phase 2)
├── train_model.py             ← Phase 3
├── requirements.txt           ← tells GitHub what libraries to install
└── .github/
    └── workflows/
        └── fetch_data.yml     ← runs fetcher automatically every week

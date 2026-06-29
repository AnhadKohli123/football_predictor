"""
⚽ Merge World Cup Data — adds international matches to your dataset
====================================================================
INSTRUCTIONS:
1. Make sure data/results.csv (from Kaggle) is in your data/ folder
2. Run:  python merge_wc.py
3. It overwrites data/all_matches.csv with the combined dataset
4. Then re-run build_features.py and train_model.py
"""

import pandas as pd
import os

KAGGLE_FILE  = "data/results.csv"
MATCHES_FILE = "data/all_matches.csv"
OUTPUT_FILE  = "data/all_matches.csv"  # overwrite with merged version

print("\n⚽ Merging World Cup data into your dataset...\n")

# ============================================================
# Check files exist
# ============================================================

for f in [KAGGLE_FILE, MATCHES_FILE]:
    if not os.path.exists(f):
        print(f"❌ Missing: {f}")
        exit()

# ============================================================
# Load existing club match data
# ============================================================

club_df = pd.read_csv(MATCHES_FILE, parse_dates=["date"])
print(f"✅ Loaded {len(club_df)} club matches from all_matches.csv")

# ============================================================
# Load and filter Kaggle international data
# ============================================================

kaggle_df = pd.read_csv(KAGGLE_FILE, parse_dates=["date"])
print(f"✅ Loaded {len(kaggle_df)} international matches from Kaggle")

# Filter to only World Cup matches
wc_df = kaggle_df[kaggle_df["tournament"] == "FIFA World Cup"].copy()
print(f"   World Cup matches found: {len(wc_df)}")

# Filter to tournaments we care about (2002 onwards)
wc_df = wc_df[wc_df["date"].dt.year >= 2002]
print(f"   World Cup matches from 2002 onwards: {len(wc_df)}")

# ============================================================
# Standardise Kaggle format to match our format
# ============================================================

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "HOME_WIN"
    elif row["home_score"] < row["away_score"]:
        return "AWAY_WIN"
    else:
        # In World Cup knockouts, draws go to penalties
        # We'll treat them as DRAW for now
        return "DRAW"

wc_standardised = pd.DataFrame({
    "league":      "World Cup",
    "date":        wc_df["date"],
    "matchday":    None,
    "home_team":   wc_df["home_team"],
    "away_team":   wc_df["away_team"],
    "home_goals":  wc_df["home_score"],
    "away_goals":  wc_df["away_score"],
    "result":      wc_df.apply(get_result, axis=1),
    "season":      wc_df["date"].dt.year.astype(str),
    "stage":       wc_df["tournament"],
})

print(f"\n📊 World Cup matches by year:")
print(wc_standardised["season"].value_counts().sort_index().to_string())

# ============================================================
# Merge both datasets
# ============================================================

combined = pd.concat([club_df, wc_standardised], ignore_index=True)
combined = combined.sort_values("date").reset_index(drop=True)
combined = combined.dropna(subset=["result"])

# Remove duplicates just in case
combined = combined.drop_duplicates(
    subset=["date", "home_team", "away_team"]
).reset_index(drop=True)

print(f"\n✅ Combined dataset:")
print(f"   Club matches:      {len(club_df)}")
print(f"   World Cup matches: {len(wc_standardised)}")
print(f"   Total:             {len(combined)}")

# ============================================================
# Save
# ============================================================

combined.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Saved to {OUTPUT_FILE}")

print(f"\n📈 Match counts by league:")
print(combined.groupby("league")["result"].count().sort_values(ascending=False).to_string())

print(f"\n🎯 Result distribution:")
print(combined["result"].value_counts().to_string())

print("""
🚀 Next steps — run these in order:
   1. python build_features.py
   2. python train_model.py
""")

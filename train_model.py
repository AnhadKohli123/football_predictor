"""
⚽ Train Model — Step 3 of your prediction model
=================================================
INSTRUCTIONS:
1. Make sure build_features.py has been run
   (you should have data/features.csv)
2. Run:  python train_model.py
3. It saves your trained AI to: models/predictor.pkl

What this script does:
- Loads your 4,945 matches with features
- Trains TWO models: Random Forest + XGBoost
- Picks the best one automatically
- Tells you the accuracy and what it learned
- Saves the model so you can use it to predict real matches
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ============================================================
# Load features
# ============================================================

INPUT_FILE  = "data/features.csv"
MODEL_DIR   = "models"
MODEL_FILE  = f"{MODEL_DIR}/predictor.pkl"
ENCODER_FILE = f"{MODEL_DIR}/label_encoder.pkl"

print("\n⚽ Training your soccer prediction AI...\n")

if not os.path.exists(INPUT_FILE):
    print("❌ Could not find data/features.csv")
    print("   Please run build_features.py first!")
    exit()

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df)} matches with {len(df.columns)} columns")

# ============================================================
# Prepare features (X) and target (y)
# ============================================================

# These are the columns the model will learn from
FEATURE_COLS = [
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

X = df[FEATURE_COLS].copy()
y = df["result"].copy()

# Fill missing h2h values with 0.33 (neutral — no history)
X = X.fillna(0.33)

# Encode result: HOME_WIN, DRAW, AWAY_WIN → 0, 1, 2
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n📊 Result distribution:")
for label, count in zip(le.classes_, np.bincount(y_encoded)):
    pct = count / len(y_encoded) * 100
    print(f"   {label}: {count} matches ({pct:.1f}%)")

# ============================================================
# Split into training and test sets
# ============================================================

# 80% train, 20% test — test set is the most recent matches
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n🔀 Split: {len(X_train)} training matches, {len(X_test)} test matches")

# ============================================================
# Train Model 1: Random Forest
# ============================================================

print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"   Accuracy: {rf_acc:.1%}")

# ============================================================
# Train Model 2: XGBoost
# ============================================================

print("\n⚡ Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss",
    verbosity=0
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print(f"   Accuracy: {xgb_acc:.1%}")

# ============================================================
# Pick the best model
# ============================================================

if xgb_acc >= rf_acc:
    best_model = xgb
    best_preds = xgb_preds
    best_name  = "XGBoost"
    best_acc   = xgb_acc
else:
    best_model = rf
    best_preds = rf_preds
    best_name  = "Random Forest"
    best_acc   = rf_acc

print(f"\n🏆 Best model: {best_name} ({best_acc:.1%} accuracy)")

# ============================================================
# Detailed results
# ============================================================

print("\n📋 Detailed breakdown:\n")
print(classification_report(
    y_test, best_preds,
    target_names=le.classes_
))

print("🔲 Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, best_preds)
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {c}" for c in le.classes_],
    columns=[f"Pred {c}" for c in le.classes_]
)
print(cm_df.to_string())

# ============================================================
# What features matter most?
# ============================================================

print("\n🧠 What your AI learned (most important features):")
if best_name == "XGBoost":
    importances = best_model.feature_importances_
else:
    importances = best_model.feature_importances_

feat_importance = pd.Series(importances, index=FEATURE_COLS)
feat_importance = feat_importance.sort_values(ascending=False)

for feat, imp in feat_importance.items():
    bar = "█" * int(imp * 100)
    print(f"   {feat:<35} {bar} {imp:.3f}")

# ============================================================
# Cross-validation (more reliable accuracy estimate)
# ============================================================

print(f"\n🔄 Cross-validation (5-fold) for reliability check...")
cv_scores = cross_val_score(best_model, X, y_encoded, cv=5, scoring="accuracy")
print(f"   Scores: {[f'{s:.1%}' for s in cv_scores]}")
print(f"   Average: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

# ============================================================
# Baseline comparison
# ============================================================

# Simplest possible model: always predict HOME_WIN
home_win_idx = list(le.classes_).index("HOME_WIN")
baseline_acc = (y_test == home_win_idx).mean()
print(f"\n📏 Baseline (always predict Home Win): {baseline_acc:.1%}")
print(f"   Your AI beats baseline by: +{(best_acc - baseline_acc):.1%}")

# ============================================================
# Save the model
# ============================================================

joblib.dump(best_model, MODEL_FILE)
joblib.dump(le, ENCODER_FILE)

print(f"\n💾 Model saved to: {MODEL_FILE}")
print(f"   Encoder saved to: {ENCODER_FILE}")
print(f"\n🚀 Next step: run  python predict.py  to predict real upcoming matches!")

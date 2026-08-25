"""
⚽ Step 3 — train the model.
============================

Reads data/features.csv, trains a Random Forest and an XGBoost model, keeps
whichever generalises better, and saves it to models/.

    python train_model.py
    python train_model.py --test-size 0.15 --seed 7

A note on how this is evaluated
-------------------------------
Matches are evaluated with a CHRONOLOGICAL split: the model trains on older
matches and is tested on the most recent ones. A random shuffled split would
let the model train on 2025 matches and be tested on 2019 ones — it would
already "know" how those teams developed, and the reported accuracy would be
optimistic in a way that never survives contact with real fixtures.

Accuracy alone is also a poor score for this problem, because predicting
"home win" every time is a surprisingly strong baseline. Log loss and the
Brier score are reported too: those judge the probabilities, which is what
you actually read off the output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, log_loss)
from sklearn.preprocessing import LabelEncoder

from features import FEATURE_COLS

INPUT_FILE = "data/features.csv"
MODEL_DIR = "models"


def multiclass_brier(y_true_onehot: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error across the predicted probability vector."""
    return float(np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1)))


def evaluate(name: str, model, X_test, y_test, classes) -> dict:
    probs = model.predict_proba(X_test)
    preds = probs.argmax(axis=1)
    onehot = np.eye(len(classes))[y_test]
    return {
        "name": name,
        "model": model,
        "preds": preds,
        "probs": probs,
        "accuracy": accuracy_score(y_test, preds),
        "log_loss": log_loss(y_test, probs, labels=list(range(len(classes)))),
        "brier": multiclass_brier(onehot, probs),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the match outcome model.")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="fraction of the most recent matches held out (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select-by", choices=["log_loss", "accuracy", "brier"],
                        default="log_loss",
                        help="metric used to pick the better model (default: log_loss)")
    args = parser.parse_args(argv)

    print("\n⚽ Training the match predictor...\n")

    if not Path(args.input).exists():
        print(f"❌ Could not find {args.input}")
        print("   Run:  python build_features.py")
        return 1

    df = pd.read_csv(args.input, parse_dates=["date"])
    df = df.dropna(subset=["result"]).sort_values("date").reset_index(drop=True)
    print(f"✅ Loaded {len(df)} matches")
    print(f"   {df['date'].min().date()} → {df['date'].max().date()}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"\n❌ features.csv is missing columns: {missing}")
        print("   Re-run build_features.py — features.py has probably changed.")
        return 1

    X = df[FEATURE_COLS].astype(float).fillna(0.33)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["result"])
    classes = list(encoder.classes_)

    print("\n📊 Result distribution:")
    for label, count in zip(classes, np.bincount(y)):
        print(f"   {label:<9} {count:>6}  ({count / len(y):.1%})")

    # ── Chronological split: train on the past, test on the future ──
    split_at = int(len(df) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y[:split_at], y[split_at:]

    print(f"\n🔀 Chronological split at {df['date'].iloc[split_at].date()}")
    print(f"   Train: {len(X_train)} matches (up to {df['date'].iloc[split_at - 1].date()})")
    print(f"   Test:  {len(X_test)} matches (from {df['date'].iloc[split_at].date()})")

    results = []

    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=15,
        class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test, classes))
    print(f"   accuracy {results[-1]['accuracy']:.1%}   log loss {results[-1]['log_loss']:.4f}")

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("\n⚠️  xgboost not installed — skipping (pip install xgboost)")
    else:
        print("\n⚡ Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.5, objective="multi:softprob", num_class=len(classes),
            random_state=args.seed, eval_metric="mlogloss", verbosity=0,
        )
        xgb.fit(X_train, y_train)
        results.append(evaluate("XGBoost", xgb, X_test, y_test, classes))
        print(f"   accuracy {results[-1]['accuracy']:.1%}   log loss {results[-1]['log_loss']:.4f}")

    # Lower is better for log loss and Brier; higher is better for accuracy.
    if args.select_by == "accuracy":
        best = max(results, key=lambda r: r["accuracy"])
    else:
        best = min(results, key=lambda r: r[args.select_by])

    print(f"\n🏆 Best by {args.select_by}: {best['name']}")
    print(f"   Accuracy  {best['accuracy']:.1%}")
    print(f"   Log loss  {best['log_loss']:.4f}   (lower is better)")
    print(f"   Brier     {best['brier']:.4f}   (lower is better)")

    # ── Baselines worth beating ──
    home_idx = classes.index("HOME_WIN") if "HOME_WIN" in classes else 0
    always_home = float((y_test == home_idx).mean())
    train_prior = np.bincount(y_train, minlength=len(classes)) / len(y_train)
    prior_probs = np.tile(train_prior, (len(y_test), 1))
    prior_ll = log_loss(y_test, prior_probs, labels=list(range(len(classes))))

    print("\n📏 Baselines:")
    print(f"   Always predict HOME_WIN     accuracy {always_home:.1%}")
    print(f"   Training-set class priors   log loss {prior_ll:.4f}")
    delta_acc = best["accuracy"] - always_home
    delta_ll = prior_ll - best["log_loss"]
    print(f"\n   Model vs. always-home:  {delta_acc:+.1%} accuracy")
    print(f"   Model vs. class priors: {delta_ll:+.4f} log loss "
          f"({'better' if delta_ll > 0 else 'WORSE — the model adds nothing'})")

    print("\n📋 Test-set breakdown:\n")
    print(classification_report(y_test, best["preds"], target_names=classes,
                                zero_division=0))

    print("🔲 Confusion matrix (rows = actual, cols = predicted):")
    cm = pd.DataFrame(
        confusion_matrix(y_test, best["preds"], labels=list(range(len(classes)))),
        index=[f"actual {c}" for c in classes],
        columns=[f"pred {c}" for c in classes],
    )
    print(cm.to_string())

    print("\n🧠 Most important features:")
    importances = pd.Series(best["model"].feature_importances_, index=FEATURE_COLS)
    for feature, value in importances.sort_values(ascending=False).head(15).items():
        bar = "█" * max(1, int(value * 120))
        print(f"   {feature:<28} {bar} {value:.3f}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], model_dir / "predictor.pkl")
    joblib.dump(encoder, model_dir / "label_encoder.pkl")

    (model_dir / "metrics.json").write_text(
        pd.Series(
            {
                "model": best["name"],
                "trained_on": len(X_train),
                "tested_on": len(X_test),
                "accuracy": round(best["accuracy"], 4),
                "log_loss": round(best["log_loss"], 4),
                "brier": round(best["brier"], 4),
                "baseline_always_home": round(always_home, 4),
                "baseline_prior_log_loss": round(prior_ll, 4),
                "n_features": len(FEATURE_COLS),
            }
        ).to_json(indent=2)
    )

    print(f"\n💾 Saved:")
    print(f"   {model_dir / 'predictor.pkl'}")
    print(f"   {model_dir / 'label_encoder.pkl'}")
    print(f"   {model_dir / 'metrics.json'}")
    print("\n🚀 Next:  python predict.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

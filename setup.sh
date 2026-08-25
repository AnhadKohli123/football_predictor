#!/usr/bin/env bash
#
# One-command setup: virtualenv, dependencies, training data, trained model.
#
#   ./setup.sh              # big five leagues, 14 seasons (recommended)
#   ./setup.sh --quick      # big five, 6 seasons — faster
#
# Safe to re-run. Skips work that is already done unless you pass --force.

set -euo pipefail

QUICK=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")"

say()  { printf '\n\033[1;32m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m !\033[0m %s\n' "$1"; }
die()  { printf '\n\033[1;31m✗\033[0m %s\n' "$1" >&2; exit 1; }

# ── Python ────────────────────────────────────────────────────────────
PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")
    major=${version%%.*}; minor=${version##*.}
    if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ]; then PY="$candidate"; break; fi
  fi
done
[ -n "$PY" ] || die "Need Python 3.9 or newer. Install it, then re-run ./setup.sh"
say "Using $($PY --version)"

# ── Virtualenv ────────────────────────────────────────────────────────
# Debian, Ubuntu and Homebrew mark the system Python as externally managed
# (PEP 668), so installing into it fails. A venv sidesteps that and keeps
# this project's dependencies off your system Python either way.
if [ ! -d .venv ] || [ "$FORCE" = "1" ]; then
  say "Creating virtualenv in .venv"
  "$PY" -m venv .venv 2>/dev/null || die \
"Could not create a virtualenv. On Debian or Ubuntu install the venv package:
    sudo apt install python3-venv
Then re-run ./setup.sh"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
say "Installing dependencies (a minute or two the first time)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
say "Dependencies installed"

# ── Training data ─────────────────────────────────────────────────────
SEASONS=14
[ "$QUICK" = "1" ] && SEASONS=6

if [ ! -f data/all_matches.csv ] || [ "$FORCE" = "1" ]; then
  say "Downloading club results (no API token needed)"
  if [ "$QUICK" = "1" ]; then
    python fetch_club_data.py --seasons "$SEASONS"
  else
    python fetch_club_data.py --all-leagues --seasons "$SEASONS"
  fi
else
  warn "data/all_matches.csv exists — skipping download (use --force to redo)"
fi

# ── Features and model ────────────────────────────────────────────────
if [ ! -f data/features.csv ] || [ "$FORCE" = "1" ]; then
  say "Building features"
  python build_features.py
else
  warn "data/features.csv exists — skipping (use --force to redo)"
fi

if [ ! -f models/predictor.pkl ] || [ "$FORCE" = "1" ]; then
  say "Training the model"
  python train_model.py
else
  warn "models/predictor.pkl exists — skipping (use --force to retrain)"
fi

# ── API token ─────────────────────────────────────────────────────────
# Only upcoming fixtures need one; everything else works offline.
if [ ! -f .env ]; then
  cat > .env <<'ENVEOF'
# Your football API token. This file is gitignored — never commit it.
# Only needed to fetch UPCOMING fixtures; predictions work without it.
# FOOTBALL_API_KEY=your_token_here
ENVEOF
  chmod 600 .env
  warn "Created .env — add your API token there to enable upcoming fixtures."
fi

say "Setup complete"
cat <<'DONE'

  Start the web app:

      source .venv/bin/activate
      python app.py

  ...then open http://127.0.0.1:5000

  Or use the terminal:

      source .venv/bin/activate
      python predict.py --match "Arsenal" "Chelsea"

DONE

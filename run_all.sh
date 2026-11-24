#!/usr/bin/env bash
# Run every experiment reported in the paper (about 30-60 minutes on a laptop CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/reproduce.py main  --protocol window
python scripts/reproduce.py main  --protocol session
python scripts/reproduce.py main  --protocol session --features no_downtime
python scripts/reproduce.py extra --protocol session
python scripts/reproduce.py pca   --protocol session

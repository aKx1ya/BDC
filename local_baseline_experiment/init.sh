#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
mkdir -p app/data app/model app/output app/temp
python - <<'PY'
from pathlib import Path

required = [
    Path("app/code/src/train.py"),
    Path("app/code/src/test.py"),
    Path("app/code/src/validate_result.py"),
    Path("app/code/src/evaluate.py"),
    Path("config.yaml"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))
print("environment initialized")
PY

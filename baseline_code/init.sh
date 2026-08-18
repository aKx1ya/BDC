#!/usr/bin/env bash
set -euo pipefail

# CUDA environment bootstrap for the overnight AutoML plan.
# Requires Python 3.11 or 3.12 on PATH as python3.11/python3.12/python.
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3.12}"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11/3.12 not found. Install it first, then rerun init.sh." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv .venv-cuda
if [ -f .venv-cuda/bin/activate ]; then
  source .venv-cuda/bin/activate
elif [ -f .venv-cuda/Scripts/activate ]; then
  source .venv-cuda/Scripts/activate
else
  echo "Could not find .venv-cuda activation script." >&2
  exit 1
fi
python -m pip install --upgrade pip
python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r <(python - <<'PY'
import tomllib
with open('pyproject.toml', 'rb') as f:
    deps = tomllib.load(f)['project']['dependencies']
for dep in deps:
    if not dep.startswith('torch'):
        print(dep)
PY
)
# Existing repository scalers were serialized with sklearn 1.7.2.
python -m pip install scikit-learn==1.7.2
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
PY

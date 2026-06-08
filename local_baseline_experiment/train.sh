#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
python app/code/src/train.py --config "${CONFIG_PATH:-config.yaml}"

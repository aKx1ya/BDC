#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
CONFIG_FILE="${CONFIG_PATH:-config.yaml}"
python app/code/src/test.py --config "$CONFIG_FILE"
RESULT_PATH=$(python - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "app/code/src")
from config import load_config, path_from_config

config = load_config(sys.argv[1])
print(path_from_config(config, "output_path") / str(config.get("result_file", "result.csv")))
PY
)
python app/code/src/validate_result.py "$RESULT_PATH"
python - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "app/code/src")
from config import load_config, path_from_config

config = load_config(sys.argv[1])
test_path = path_from_config(config, "data_path") / str(config.get("test_file", "test.csv"))
if test_path.exists():
    result_path = path_from_config(config, "output_path") / str(config.get("result_file", "result.csv"))
    temp_path = path_from_config(config, "temp_path")
    import subprocess
    subprocess.run(
        [
            sys.executable,
            "app/code/src/evaluate.py",
            "--result",
            str(result_path),
            "--test",
            str(test_path),
            "--temp",
            str(temp_path),
        ],
        check=True,
    )
else:
    print(f"skip evaluation: {test_path} not found")
PY

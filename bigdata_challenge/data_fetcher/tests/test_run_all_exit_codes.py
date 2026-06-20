from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_all.py"


def load_module():
    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("run_all", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_single_step_failure_returns_non_zero_exit_code(monkeypatch) -> None:
    module = load_module()

    monkeypatch.setattr(sys, "argv", ["run_all.py", "--step", "1"])
    monkeypatch.setattr(module, "run_step", lambda step: False)

    assert module.main() == 1


def test_single_step_success_returns_zero_exit_code(monkeypatch) -> None:
    module = load_module()

    monkeypatch.setattr(sys, "argv", ["run_all.py", "--step", "1"])
    monkeypatch.setattr(module, "run_step", lambda step: True)

    assert module.main() == 0

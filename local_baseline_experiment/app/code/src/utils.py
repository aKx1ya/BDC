from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def stable_rank_desc(values: Iterable[float]) -> np.ndarray:
    series = np.asarray(list(values), dtype=float)
    order = np.lexsort((np.arange(len(series)), -series))
    ranks = np.empty(len(series), dtype=int)
    ranks[order] = np.arange(1, len(series) + 1)
    return ranks


def file_md5(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_stock_id(value: object) -> str:
    text = str(value).strip()
    if text.startswith(("sh.", "sz.")):
        text = text.split(".", 1)[1]
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)
    return text


def format_stock_id_for_output(value: object) -> str:
    return canonical_stock_id(value)

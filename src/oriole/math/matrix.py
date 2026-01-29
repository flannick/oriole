from __future__ import annotations

from typing import Callable

import numpy as np


def matrix_fill(n_rows: int, n_cols: int, func: Callable[[int, int], float]) -> np.ndarray:
    data = np.empty((n_rows, n_cols), dtype=float)
    for i in range(n_rows):
        for j in range(n_cols):
            data[i, j] = func(i, j)
    return data


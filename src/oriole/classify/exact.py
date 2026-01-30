from __future__ import annotations

import numpy as np

from ..params import Params


def calculate_mu(params: Params, betas: list[float], ses: list[float]) -> np.ndarray:
    from .analytical import calculate_mu_vec

    return calculate_mu_vec(params, betas, ses)

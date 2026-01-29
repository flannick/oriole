from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..data import GwasData, Meta
from ..params import Params


@dataclass
class Vars:
    meta: Meta
    es: np.ndarray
    ts: np.ndarray

    def indices(self) -> Iterator[tuple[str, int, int | None]]:
        n_data_points = self.meta.n_data_points()
        n_traits = self.meta.n_traits()
        for i_data_point in range(n_data_points):
            yield ("e", i_data_point, None)
            for i_trait in range(n_traits):
                yield ("t", i_data_point, i_trait)

    @staticmethod
    def initial_vars(data: GwasData, params: Params) -> "Vars":
        es = np.full(data.n_data_points(), params.mu, dtype=float)
        betas = np.asarray(params.betas, dtype=float)
        ts = es[:, None] * betas[None, :]
        return Vars(meta=data.meta, es=es, ts=ts)

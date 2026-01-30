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
            for i_endo in range(self.es.shape[1]):
                yield ("e", i_data_point, i_endo)
            for i_trait in range(n_traits):
                yield ("t", i_data_point, i_trait)

    @staticmethod
    def initial_vars(data: GwasData, params: Params) -> "Vars":
        n_data_points = data.n_data_points()
        mus = np.asarray(params.mus, dtype=float)
        betas = np.asarray(params.betas, dtype=float)
        es = np.tile(mus[None, :], (n_data_points, 1))
        ts = es @ betas.T
        return Vars(meta=data.meta, es=es, ts=ts)

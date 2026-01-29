from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from ..data import GwasData, Meta
from ..params import Params
from .gibbs import GibbsSampler
from .var_stats import VarStats
from .vars import Vars


class ETracer(Protocol):
    def trace_e(self, e: float) -> None: ...


class Sampler:
    def __init__(self, meta: Meta, rng: np.random.Generator) -> None:
        self._gibbs = GibbsSampler(rng)
        self._var_stats = VarStats(meta)

    def sample_n(
        self,
        data: GwasData,
        params: Params,
        vars: Vars,
        n_steps: int,
        e_tracer: Optional[ETracer],
        t_pinned: bool,
    ) -> None:
        for _ in range(n_steps):
            self.sample_one(data, params, vars, e_tracer, t_pinned)

    def sample_one(
        self,
        data: GwasData,
        params: Params,
        vars: Vars,
        e_tracer: Optional[ETracer],
        t_pinned: bool,
    ) -> None:
        for var_type, i_data_point, i_trait in vars.indices():
            if var_type == "e":
                e = self._gibbs.draw_e(vars, params, i_data_point)
                if e_tracer is not None:
                    e_tracer.trace_e(e)
                vars.es[i_data_point] = e
            else:
                vars.ts[i_data_point, i_trait] = self._gibbs.draw_t(
                    data, vars, params, i_data_point, i_trait, t_pinned
                )
        self._var_stats.add(vars)

    @property
    def var_stats(self) -> VarStats:
        return self._var_stats


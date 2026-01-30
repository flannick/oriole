from __future__ import annotations

import numpy as np

from ..data import LoadedData
from ..options.config import TrainConfig
from ..params import Params
from ..sample.sampler import Sampler
from ..sample.vars import Vars


class MessageToWorker:
    def __init__(self, kind: str, payload=None):
        self.kind = kind
        self.payload = payload

    @classmethod
    def take_n_samples(cls, n_samples: int) -> "MessageToWorker":
        return cls("take_n_samples", n_samples)

    @classmethod
    def set_new_params(cls, params: Params) -> "MessageToWorker":
        return cls("set_new_params", params)

    @classmethod
    def shutdown(cls) -> "MessageToWorker":
        return cls("shutdown")


class MessageToCentral:
    def __init__(self, i_thread: int, params: Params) -> None:
        self._i_thread = i_thread
        self.params = params

    def i_thread(self) -> int:
        return self._i_thread


class TrainWorkerLauncher:
    def __init__(
        self, data: LoadedData, params: Params, config: TrainConfig, mask: np.ndarray
    ) -> None:
        self.data = data
        self.params = params
        self.config = config
        self.mask = mask

    def launch(self, in_queue, out_queue, i_thread: int) -> None:
        train_worker(
            self.data, self.params, in_queue, out_queue, i_thread, self.config, self.mask
        )


def train_worker(
    data: LoadedData,
    params: Params,
    in_queue,
    out_queue,
    i_thread: int,
    config: TrainConfig,
    mask: np.ndarray,
) -> None:
    vars = Vars.initial_vars(data.gwas_data, params)
    rng = np.random.default_rng()
    meta = data.gwas_data.meta
    sampler = Sampler(meta, rng)
    t_pinned = bool(config.t_pinned) if config.t_pinned is not None else False
    sampler.sample_n(data.gwas_data, params, vars, config.n_steps_burn_in, None, t_pinned)
    sampler.reset_stats()

    while True:
        message: MessageToWorker = out_queue.get()
        if message.kind == "take_n_samples":
            sampler.reset_stats()
            sampler.sample_n(data.gwas_data, params, vars, int(message.payload), None, t_pinned)
            params_new = sampler.var_stats.compute_new_params(data.weights, mask)
            in_queue.put(MessageToCentral(i_thread, params_new))
        elif message.kind == "set_new_params":
            params = message.payload
            sampler.sample_n(data.gwas_data, params, vars, config.n_steps_burn_in, None, t_pinned)
            sampler.reset_stats()
        elif message.kind == "shutdown":
            break

from __future__ import annotations

import numpy as np

from .analytical import analytical_classification, calculate_mu_vec
from ..data import GwasData
from ..options.config import ClassifyConfig
from ..params import Params
from ..sample.sampler import Sampler, ETracer
from ..sample.vars import Vars


class MessageToWorker:
    def __init__(self, kind: str, payload=None) -> None:
        self.kind = kind
        self.payload = payload

    @classmethod
    def data_point(cls, i_data_point: int) -> "MessageToWorker":
        return cls("data_point", i_data_point)

    @classmethod
    def shutdown(cls) -> "MessageToWorker":
        return cls("shutdown")


class Classification:
    def __init__(self, sampled, e_mean_calculated: float) -> None:
        self.sampled = sampled
        self.e_mean_calculated = e_mean_calculated


class MessageToCentral:
    def __init__(self, i_thread: int, classification: Classification) -> None:
        self._i_thread = i_thread
        self.classification = classification

    def i_thread(self) -> int:
        return self._i_thread


class ClassifyETracer(ETracer):
    def __init__(self, handle) -> None:
        self._handle = handle

    def trace_e(self, e_values) -> None:
        try:
            values = "\t".join(str(float(value)) for value in e_values)
            self._handle.write(f"{values}\n")
        except Exception as exc:
            print(f"Could not write E trace: {exc}")


class ClassifyWorkerLauncher:
    def __init__(
        self,
        data: GwasData,
        params: Params,
        config: ClassifyConfig,
        analytical: bool = True,
    ) -> None:
        self.data = data
        self.params = params
        self.config = config
        self.analytical = analytical

    def launch(self, in_queue, out_queue, i_thread: int) -> None:
        classify_worker(
            self.data,
            self.params,
            self.config,
            in_queue,
            out_queue,
            i_thread,
            analytical=self.analytical,
        )


def classify_worker(
    data: GwasData,
    params: Params,
    config: ClassifyConfig,
    in_queue,
    out_queue,
    i_thread: int,
    analytical: bool = True,
) -> None:
    while True:
        message: MessageToWorker = out_queue.get()
        if message.kind == "data_point":
            i_data_point = int(message.payload)
            data_point, is_col = data.only_data_point(i_data_point)
            params_reduced = params.reduce_to(data_point.meta.trait_names, is_col)
            vars = Vars.initial_vars(data_point, params_reduced)
            rng = np.random.default_rng()
            if analytical:
                sampled = analytical_classification(
                    params_reduced, list(data_point.betas[0]), list(data_point.ses[0])
                )
            else:
                sampler = Sampler(data_point.meta, rng)
                e_tracer = None
                trace_handle = None
                if config.trace_ids and data_point.meta.var_ids:
                    var_id = data_point.meta.var_ids[0]
                    if var_id in config.trace_ids:
                        trace_file_name = f"{config.out_file}_{var_id}"
                        try:
                            trace_handle = open(trace_file_name, "w", encoding="utf-8")
                            e_tracer = ClassifyETracer(trace_handle)
                        except Exception as exc:
                            print(f"Could not create E trace file: {exc}")
                            e_tracer = None
                t_pinned = bool(config.t_pinned) if config.t_pinned is not None else False
                sampler.sample_n(
                    data_point,
                    params_reduced,
                    vars,
                    config.n_steps_burn_in,
                    e_tracer,
                    t_pinned,
                )
                sampler.reset_stats()
                sampler.sample_n(
                    data_point,
                    params_reduced,
                    vars,
                    config.n_samples,
                    e_tracer,
                    t_pinned,
                )
                sampled = sampler.var_stats.calculate_classification()
            mu_calculated = calculate_mu_vec(
                params_reduced,
                list(data_point.betas[0]),
                list(data_point.ses[0]),
            )
            classification = Classification(sampled=sampled, e_mean_calculated=mu_calculated)
            in_queue.put(MessageToCentral(i_thread, classification))
            if not analytical:
                if trace_handle is not None:
                    trace_handle.close()
        elif message.kind == "shutdown":
            break

from __future__ import annotations

import os
from pathlib import Path

from ..data import load_data
from ..error import new_error
from ..options.action import Action
from ..options.config import Config
from ..params import Params, write_params_to_file
from ..report import Reporter
from ..sample.trace_file import ParamTraceFileWriter
from ..train.initial_params import estimate_initial_params
from ..train.param_meta_stats import ParamMetaStats
from ..util.threads import Threads
from .worker import MessageToWorker, TrainWorkerLauncher


def train_or_check(config: Config, dry: bool, match_rust: bool = False) -> None:
    data = load_data(config, Action.TRAIN)
    print(f"Loaded data for {data.gwas_data.meta.n_data_points()} variants")
    print(data.gwas_data)
    if dry:
        print("User picked dry run only, so doing nothing.")
        return
    train(data, config, match_rust=match_rust)


def train(data, config: Config, match_rust: bool = False) -> None:
    n_traits = data.gwas_data.meta.n_traits()
    params_trace_writer = None
    if config.files.trace:
        params_trace_writer = ParamTraceFileWriter(Path(config.files.trace), n_traits)

    n_threads = max(os.cpu_count() or 1, 3)
    print(
        f"Launching {n_threads} workers and burning in with {config.train.n_steps_burn_in} iterations"
    )
    params = estimate_initial_params(data.gwas_data, match_rust=match_rust)
    print(params)

    launcher = TrainWorkerLauncher(data, params, config.train)
    threads = Threads.new(launcher, n_threads)
    print("Workers launched and burned in.")

    n_samples = config.train.n_samples_per_iteration
    reporter = Reporter()
    i_round = 0
    i_iteration = 0
    while True:
        params0 = create_param_estimates(threads, n_samples)
        params1 = create_param_estimates(threads, n_samples)
        param_meta_stats = ParamMetaStats(
            n_threads, params.trait_names, params0, params1
        )
        reached_precision = False
        while True:
            i_iteration += 1
            params_new = create_param_estimates(threads, n_samples)
            param_meta_stats.add(params_new)
            summary = param_meta_stats.summary()
            if i_iteration >= config.train.n_iterations_per_round:
                params = summary.params
                if params_trace_writer is not None:
                    params_trace_writer.write(params)
                if i_round >= config.train.n_rounds:
                    print("Done!")
                    reached_precision = True
                else:
                    i_round += 1
                    print(
                        f"Setting new parameters for round {i_round} after {i_iteration} iterations"
                    )
                    i_iteration = 0
                    for out_queue in threads.out_queues:
                        out_queue.put(MessageToWorker.set_new_params(params))
                reporter.report(summary, i_round, i_iteration, n_samples)
                reporter.reset_round_timer()
                break
        if reached_precision:
            break

    if config.train.normalize_mu_to_one:
        params = params.normalized_with_mu_one()
    write_params_to_file(params, config.files.params)
    threads.close(MessageToWorker.shutdown())


def create_param_estimates(threads: Threads, n_samples: int) -> list[Params]:
    threads.broadcast(MessageToWorker.take_n_samples(n_samples))
    responses = threads.responses_from_all()
    return [response.params for response in responses]

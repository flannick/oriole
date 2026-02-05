from __future__ import annotations

import os
import time
from pathlib import Path
import numpy as np

from ..data import load_data, load_ids, LoadedData, Weights
from ..error import new_error
from ..options.action import Action
from ..options.config import Config, endophenotype_mask
from ..options.inference import resolve_inference, build_outlier_pis
from ..params import Params, write_params_to_file
from ..report import Reporter
from ..sample.trace_file import ParamTraceFileWriter
from ..train.initial_params import estimate_initial_params
from ..train.analytical import estimate_params_analytical, log_marginal_likelihood
from ..train.outliers_analytic import estimate_params_analytical_outliers
from ..train.outliers_variational import estimate_params_variational_outliers
from ..train.tune_outliers import build_tune_cache, tune_outliers
from ..train.diagnostics import plot_param_convergence, plot_cv_grid
from ..train.param_meta_stats import ParamMetaStats
from ..util.threads import Threads
from .worker import MessageToWorker, TrainWorkerLauncher


class SummaryStub:
    def __init__(self, params: Params) -> None:
        self.params = params
        self.n_chains_used = 1
        self.intra_chain_vars = []
        self.inter_chain_vars = []


def _params_vector(params: Params) -> np.ndarray:
    parts = [
        np.asarray(params.mus, dtype=float).ravel(),
        np.asarray(params.taus, dtype=float).ravel(),
        np.asarray(params.betas, dtype=float).ravel(),
        np.asarray(params.sigmas, dtype=float).ravel(),
        np.asarray(params.trait_edges, dtype=float).ravel(),
        np.asarray([params.outlier_kappa], dtype=float),
        np.asarray(params.outlier_pis, dtype=float).ravel(),
    ]
    return np.concatenate(parts)


def train_or_check(
    config: Config,
    dry: bool,
    match_rust: bool = False,
    inference: str = "auto",
    chunk_size: int | None = None,
    verbose: bool = False,
) -> None:
    n_traits = len(config.gwas)
    inference_mode = resolve_inference(config, inference, n_traits)
    tune_cache = None
    if config.tune_outliers.enabled:
        cache = build_tune_cache(config)
        tune_cache = cache
        weight_map = load_ids(config.train.ids_file, n_traits)
        weights_list: list[float] = []
        total_weight = 0.0
        for var_id in cache.pos_data.meta.var_ids:
            weight = weight_map.get(var_id)
            value = 1.0 if weight is None else float(weight.weight)
            weights_list.append(value)
            total_weight += value
        data = LoadedData(
            gwas_data=cache.pos_data,
            weights=Weights(weights=weights_list, sum=total_weight),
        )
        print(
            "Using cached tuning data; "
            f"training variants={data.gwas_data.meta.n_data_points()}"
        )
    else:
        load_start = time.perf_counter()
        data = load_data(config, Action.TRAIN)
        print(f"Loaded training data in {time.perf_counter() - load_start:.2f}s")
    if verbose:
        print(
            "Train config: {} traits, {} endos, {} edges, analytical={}, chunk_size={}".format(
                n_traits,
                len(config.endophenotypes),
                len(config.trait_edges),
                inference_mode,
                chunk_size or "auto",
            )
        )
    print(f"Loaded data for {data.gwas_data.meta.n_data_points()} variants")
    print(data.gwas_data)
    if dry:
        print("User picked dry run only, so doing nothing.")
        return
    train(
        data,
        config,
        match_rust=match_rust,
        inference=inference_mode,
        chunk_size=chunk_size,
        tune_cache=tune_cache,
    )


def _default_chunk_size(n_traits: int, n_data_points: int) -> int:
    bytes_target = 2 * 1024 ** 3
    bytes_per_variant = (n_traits * 8 * 6) + (8 * 3)
    chunk = max(1, bytes_target // bytes_per_variant)
    if n_data_points > 0:
        chunk = min(chunk, n_data_points)
    return chunk


def train(
    data,
    config: Config,
    match_rust: bool = False,
    inference: str = "auto",
    chunk_size: int | None = None,
    tune_cache=None,
) -> None:
    n_traits = data.gwas_data.meta.n_traits()
    trace_path: Path | None = None
    if config.files.trace:
        trace_path = Path(config.files.trace)
    elif config.train.plot_convergence_out_file:
        trace_name = Path(config.train.plot_convergence_out_file).stem + ".trace.tsv"
        trace_path = Path("results") / trace_name
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    params_trace_writer = None
    if trace_path is not None:
        params_trace_writer = ParamTraceFileWriter(
            trace_path, n_traits, len(config.endophenotypes)
        )

    n_threads = max(os.cpu_count() or 1, 3)
    print(
        f"Launching {n_threads} workers and burning in with {config.train.n_steps_burn_in} iterations"
    )
    mask = np.asarray(endophenotype_mask(config), dtype=bool)
    trait_names = [item.name for item in config.gwas]
    trait_index = {name: idx for idx, name in enumerate(trait_names)}
    parent_mask = np.zeros((len(trait_names), len(trait_names)), dtype=bool)
    for edge in config.trait_edges:
        parent_mask[trait_index[edge.child], trait_index[edge.parent]] = True
    should_tune = (
        config.tune_outliers.enabled
        and config.tune_outliers.mode != "off"
    )
    if should_tune:
        if not config.outliers.enabled:
            config.outliers.enabled = True
        tune_start = time.perf_counter()
        result = tune_outliers(config, inference, cache=tune_cache)
        print(f"CV tuning phase took {time.perf_counter() - tune_start:.2f}s")
        config.outliers.kappa = result.kappa
        config.outliers.pi = result.pi
        config.outliers.expected_outliers = result.expected_outliers
        config.outliers.pi_by_trait = None
        if config.train.plot_cv_out_file:
            plot_cv_grid(
                result.scores,
                result.kappa_grid,
                result.expected_outliers_grid,
                (result.kappa, result.expected_outliers),
                config.train.plot_cv_out_file,
            )
    elif config.outliers.enabled and not (
        config.outliers.kappa_specified
        or config.outliers.pi_specified
        or config.outliers.expected_outliers_specified
    ):
        config.outliers.pi_by_trait = None
    params = estimate_initial_params(
        data.gwas_data, config.endophenotypes, mask, match_rust=match_rust
    )
    if config.outliers.enabled:
        params.outlier_kappa = config.outliers.kappa
        params.outlier_pis = build_outlier_pis(config, params.trait_names)
    print(params)

    if inference in ("analytic", "variational"):
        if chunk_size is None or chunk_size <= 0:
            chunk_size = _default_chunk_size(
                data.gwas_data.meta.n_traits(), data.gwas_data.meta.n_data_points()
            )
        print(
            "Analytical/variational training: "
            f"traits={data.gwas_data.meta.n_traits()} "
            f"variants={data.gwas_data.meta.n_data_points()} "
            f"chunk_size={chunk_size}"
        )
        total_iters = max(1, config.train.n_rounds)
        report_every = max(1, config.train.n_iterations_per_round)
        reporter = Reporter()
        total_time = 0.0
        prev_vec: np.ndarray | None = None
        prev_obj: float | None = None
        rel_ok = 0
        obj_ok = 0
        for i_iteration in range(total_iters):
            iter_start = time.perf_counter()
            if config.outliers.enabled:
                if inference == "analytic":
                    params = estimate_params_analytical_outliers(
                        data, params, chunk_size, mask, parent_mask
                    )
                else:
                    params = estimate_params_variational_outliers(
                        data, params, chunk_size, mask, parent_mask
                    )
            else:
                params = estimate_params_analytical(data, params, chunk_size, mask, parent_mask)
            iter_elapsed = time.perf_counter() - iter_start
            total_time += iter_elapsed
            print(
                f"Iteration {i_iteration + 1}/{total_iters} "
                f"took {iter_elapsed:.2f}s (cumulative {total_time:.2f}s)"
            )
            if config.train.early_stop:
                vec = _params_vector(params)
                if prev_vec is not None:
                    denom = np.maximum(np.abs(prev_vec), 1e-12)
                    rel_change = float(np.max(np.abs(vec - prev_vec) / denom))
                    if rel_change < config.train.early_stop_rel_tol:
                        rel_ok += 1
                    else:
                        rel_ok = 0
                    print(
                        f"Early stop check: rel_change={rel_change:.3e} "
                        f"(patience {rel_ok}/{config.train.early_stop_patience})"
                    )
                prev_vec = vec

                obj_available = not config.outliers.enabled
                if obj_available:
                    obj = log_marginal_likelihood(data, params, chunk_size)
                    if prev_obj is not None:
                        denom = max(abs(prev_obj), 1e-12)
                        obj_change = abs(obj - prev_obj) / denom
                        if obj_change < config.train.early_stop_obj_tol:
                            obj_ok += 1
                        else:
                            obj_ok = 0
                        print(
                            f"Early stop check: obj_change={obj_change:.3e} "
                            f"(patience {obj_ok}/{config.train.early_stop_patience})"
                        )
                    prev_obj = obj
                else:
                    print(
                        "Early stop objective check skipped (outliers enabled); "
                        "using parameter change only."
                    )

                if i_iteration + 1 >= config.train.early_stop_min_iters:
                    rel_ready = rel_ok >= config.train.early_stop_patience
                    obj_ready = obj_ok >= config.train.early_stop_patience
                    if obj_available:
                        if rel_ready and obj_ready:
                            print("Early stopping: both criteria satisfied.")
                            break
                    else:
                        if rel_ready:
                            print("Early stopping: parameter criterion satisfied.")
                            break
            if params_trace_writer is not None:
                params_trace_writer.write(params)
            if (i_iteration + 1) % report_every == 0 or i_iteration == total_iters - 1:
                reporter.report(
                    SummaryStub(params),
                    i_iteration // report_every,
                    (i_iteration + 1),
                    0,
                )
    if config.train.normalize_mu_to_one:
        params = params.normalized_with_mu_one()
    write_params_to_file(params, config.files.params)
    if config.train.plot_convergence_out_file and trace_path is not None:
        plot_param_convergence(str(trace_path), config.train.plot_convergence_out_file)
    return

    launcher = TrainWorkerLauncher(data, params, config.train, mask, parent_mask)
    threads = Threads.new(launcher, n_threads)
    print("Workers launched and burned in.")

    n_samples = config.train.n_samples_per_iteration
    if n_samples <= 0:
        n_samples = min(100, data.gwas_data.meta.n_data_points())
    reporter = Reporter()
    i_round = 0
    i_iteration = 0
    while True:
        params0 = create_param_estimates(threads, n_samples)
        params1 = create_param_estimates(threads, n_samples)
        param_meta_stats = ParamMetaStats(
            n_threads, params.trait_names, params.endo_names, params0, params1
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
    if config.train.plot_convergence_out_file and trace_path is not None:
        plot_param_convergence(str(trace_path), config.train.plot_convergence_out_file)
    threads.close(MessageToWorker.shutdown())


def create_param_estimates(threads: Threads, n_samples: int) -> list[Params]:
    threads.broadcast(MessageToWorker.take_n_samples(n_samples))
    responses = threads.responses_from_all()
    return [response.params for response in responses]

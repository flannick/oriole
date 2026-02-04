from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..check import check_params
from ..data import GwasData, load_data, Meta
from ..error import new_error
from ..options.action import Action
from ..options.config import ClassifyConfig, Config
from ..options.inference import resolve_inference
from ..params import Params, read_params_from_file
from ..sample.var_stats import SampledClassification
from ..util.threads import Threads, TaskQueueObserver
from .worker import Classification, MessageToCentral, MessageToWorker, ClassifyWorkerLauncher
from .analytical import analytical_classification_chunk, calculate_mu_chunk
from .gibbs_vectorized import gibbs_classification_chunk
from .outliers_analytic import outliers_analytic_classification_chunk
from .outliers_variational import outliers_variational_classification_chunk


class Observer(TaskQueueObserver):
    def __init__(self, var_ids: list[str], out_file: str, meta: Meta) -> None:
        self.meta = meta
        self.var_ids = var_ids
        self.out_file = out_file
        self._writer = open(out_file, "w", encoding="utf-8")

    def going_to_start_queue(self) -> None:
        print("Starting to classify data points.")
        self._writer.write(self._header())

    def going_to_send(self, out_message, i_task: int, i_thread: int) -> None:
        if out_message.kind == "shutdown":
            print(f"Sent shutdown as task {i_task} to thread {i_thread}")

    def have_received(self, in_message: MessageToCentral, i_task: int, i_thread: int) -> None:
        var_id = self.var_ids[i_task]
        try:
            self._writer.write(format_entry(var_id, in_message.classification))
        except Exception as exc:
            print(f"Cannot write temp file: {exc}")

    def nothing_more_to_send(self) -> None:
        print("No more data points to add to queue.")

    def completed_queue(self) -> None:
        print("Completed classification of all data points.")
        self._writer.close()

    def _header(self) -> str:
        return format_header(self.meta)


def classify_or_check(
    config: Config,
    dry: bool,
    inference: str = "auto",
    chunk_size: int | None = None,
    verbose: bool = False,
) -> None:
    params = read_params_from_file(config.files.params)
    check_params(config, params)
    inference_mode = resolve_inference(config, inference, len(config.gwas))
    if verbose:
        print(
            "Classify config: {} traits, {} endos, {} edges, inference={}, chunk_size={}".format(
                len(config.gwas),
                len(config.endophenotypes),
                len(config.trait_edges),
                inference_mode,
                chunk_size or "auto",
            )
        )
    print(f"Read from file mus = {params.mus}, taus = {params.taus}")
    if config.classify.params_override is not None:
        params = params.plus_overwrite(config.classify.params_override)
        print(f"After overwrite, mus = {params.mus}, taus = {params.taus}")
    data = load_data(config, Action.CLASSIFY)
    if dry:
        print("User picked dry run only, so doing nothing.")
        return
    classify(
        data.gwas_data,
        params,
        config,
        inference=inference_mode,
        chunk_size=chunk_size,
    )


def _default_chunk_size(n_traits: int, n_data_points: int) -> int:
    bytes_target = 2 * 1024 ** 3
    bytes_per_variant = (n_traits * 8 * 6) + (8 * 3)
    chunk = max(1, bytes_target // bytes_per_variant)
    if n_data_points > 0:
        chunk = min(chunk, n_data_points)
    return chunk


def classify(
    data: GwasData,
    params: Params,
    config: Config,
    inference: str = "auto",
    chunk_size: int | None = None,
) -> None:
    classify_config = config.classify
    if classify_config.n_samples <= 0:
        classify_config = ClassifyConfig(
            params_override=classify_config.params_override,
            n_steps_burn_in=classify_config.n_steps_burn_in,
            n_samples=min(1000, data.meta.n_data_points()),
            out_file=classify_config.out_file,
            trace_ids=classify_config.trace_ids,
            t_pinned=classify_config.t_pinned,
        )
    if chunk_size is None or chunk_size <= 0:
        chunk_size = _default_chunk_size(data.meta.n_traits(), data.meta.n_data_points())
    use_vectorized = chunk_size > 1 and inference == "analytic" and not config.outliers.enabled
    if use_vectorized:
        classify_vectorized(data, params, classify_config, inference, chunk_size)
        return

    n_threads = max(os.cpu_count() or 1, 3)
    launcher = ClassifyWorkerLauncher(
        data, params, classify_config, inference=inference, outliers_enabled=config.outliers.enabled
    )
    threads = Threads.new(launcher, n_threads)
    meta = data.meta
    out_messages = (MessageToWorker.data_point(i) for i in range(meta.n_data_points()))
    temp_out_file = f"{classify_config.out_file}_tmp"
    observer = Observer(meta.var_ids, temp_out_file, meta)
    in_messages = threads.task_queue(out_messages, observer)
    classifications = [message.classification for message in in_messages]
    write_out_file(classify_config.out_file, meta, classifications)
    threads.close(MessageToWorker.shutdown())


def classify_vectorized(
    data: GwasData,
    params: Params,
    config: ClassifyConfig,
    inference: str,
    chunk_size: int,
) -> None:
    if inference != "analytic":
        raise new_error("Vectorized classification is only supported for analytic inference.")
    meta = data.meta
    n = meta.n_data_points()
    with open(config.out_file, "w", encoding="utf-8") as handle:
        handle.write(format_header(meta))
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            betas = data.betas[start:end, :]
            ses = data.ses[start:end, :]
            if np.isnan(betas).any() or np.isnan(ses).any():
                for i in range(start, end):
                    single, is_col = data.only_data_point(i)
                    params_reduced = params.reduce_to(single.meta.trait_names, is_col)
                    if inference == "analytic":
                        sampled = analytical_classification_chunk(
                            params_reduced, single.betas, single.ses
                        )
                    else:
                        sampled = gibbs_classification_chunk(
                            params_reduced,
                            single.betas,
                            single.ses,
                            config.n_steps_burn_in,
                            config.n_samples,
                            config.t_pinned or False,
                        )
                    if inference == "analytic":
                        mu_calc = calculate_mu_chunk(
                            params_reduced,
                            single.betas,
                            single.ses,
                        )[0]
                    else:
                        mu_calc = sampled.e_mean[0]
                    classification = Classification(sampled, mu_calc)
                    handle.write(format_entry(single.meta.var_ids[0], classification))
                continue

            if inference == "analytic":
                sampled = analytical_classification_chunk(params, betas, ses)
            elif inference == "variational":
                sampled = outliers_variational_classification_chunk(params, betas, ses)
            else:
                sampled = outliers_analytic_classification_chunk(params, betas, ses)
            if inference == "analytic":
                mu_calc = calculate_mu_chunk(params, betas, ses)
            else:
                mu_calc = sampled.e_mean
            for i, var_id in enumerate(meta.var_ids[start:end]):
                classification = Classification(
                    SampledClassification(
                        e_mean=sampled.e_mean[i],
                        e_std=sampled.e_std[i],
                        t_means=sampled.t_means[i],
                    ),
                    mu_calc[i],
                )
                handle.write(format_entry(var_id, classification))


def write_out_file(file: str, meta: Meta, classifications: list[Classification]) -> None:
    with open(file, "w", encoding="utf-8") as handle:
        handle.write(format_header(meta))
        for var_id, classification in zip(meta.var_ids, classifications):
            handle.write(format_entry(var_id, classification))


def format_header(meta: Meta) -> str:
    parts = ["id"]
    for endo in meta.endo_names:
        parts.append(f"{endo}_mean_samp")
        parts.append(f"{endo}_std_samp")
        parts.append(f"{endo}_mean_calc")
    parts.extend(meta.trait_names)
    return "\t".join(parts) + "\n"


def format_entry(var_id: str, classification: Classification) -> str:
    sampled: SampledClassification = classification.sampled
    e_mean = np.asarray(sampled.e_mean, dtype=float)
    e_std = np.asarray(sampled.e_std, dtype=float)
    if e_mean.ndim == 2:
        e_mean = e_mean[0]
    if e_std.ndim == 2:
        e_std = e_std[0]
    e_calc = np.asarray(classification.e_mean_calculated, dtype=float)
    parts = [var_id]
    for idx in range(len(e_mean)):
        parts.append(str(float(e_mean[idx])))
        parts.append(str(float(e_std[idx])))
        parts.append(str(float(e_calc[idx])))
    t_means = np.asarray(sampled.t_means, dtype=float)
    if t_means.ndim == 2:
        t_means = t_means[0]
    parts.extend(str(float(value)) for value in t_means)
    return "\t".join(parts) + "\n"

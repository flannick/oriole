from __future__ import annotations

import gzip
import io
import math
import os
from dataclasses import dataclass

import numpy as np

from ..check import check_params
from ..data import GwasData, VariantMeta, load_data, Meta
from ..error import new_error
from ..options.action import Action
from ..options.config import ClassifyConfig, Config
from ..options.inference import resolve_inference
from ..params import Params, ParamsOverride, read_params_from_file
from ..sample.var_stats import SampledClassification
from ..util.threads import Threads, TaskQueueObserver
from .worker import Classification, MessageToCentral, MessageToWorker, ClassifyWorkerLauncher
from .analytical import (
    analytical_classification_chunk,
    calculate_mu_chunk,
    gls_endophenotype_stats_chunk,
)
from .gibbs_vectorized import gibbs_classification_chunk
from .outliers_analytic import outliers_analytic_classification_chunk, outliers_analytic_gls_chunk
from .outliers_variational import (
    outliers_variational_classification_chunk,
    outliers_variational_gls_chunk,
)


class Observer(TaskQueueObserver):
    def __init__(self, var_ids: list[str], out_file: str | None, meta: Meta) -> None:
        self.meta = meta
        self.var_ids = var_ids
        self.out_file = out_file
        self._writer = _open_output(out_file) if out_file else None

    def going_to_start_queue(self) -> None:
        print("Starting to classify data points.")
        if self._writer:
            self._writer.write(self._header())

    def going_to_send(self, out_message, i_task: int, i_thread: int) -> None:
        if out_message.kind == "shutdown":
            print(f"Sent shutdown as task {i_task} to thread {i_thread}")

    def have_received(self, in_message: MessageToCentral, i_task: int, i_thread: int) -> None:
        var_id = self.var_ids[i_task]
        try:
            if self._writer:
                self._writer.write(format_entry(var_id, in_message.classification))
        except Exception as exc:
            print(f"Cannot write temp file: {exc}")

    def nothing_more_to_send(self) -> None:
        print("No more data points to add to queue.")

    def completed_queue(self) -> None:
        print("Completed classification of all data points.")
        if self._writer:
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
    if config.classify.mu_specified or config.classify.tau_specified:
        print(
            "Warning: classify mu/tau specified in config. "
            "These override recommended defaults (mu=0, tau=1e6)."
        )
    if config.classify.params_override is not None:
        params = params.plus_overwrite(config.classify.params_override)
        print(f"After overwrite, mus = {params.mus}, taus = {params.taus}")
    else:
        params = params.plus_overwrite(
            ParamsOverride(
                mus=[config.classify.mu for _ in params.endo_names],
                taus=[config.classify.tau for _ in params.endo_names],
            )
        )
        print(f"After default override, mus = {params.mus}, taus = {params.taus}")
    data = load_data(config, Action.CLASSIFY)
    if dry:
        print("User picked dry run only, so doing nothing.")
        return
    classify(
        data.gwas_data,
        params,
        config,
        variant_meta=data.variant_meta,
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
    variant_meta: dict[str, VariantMeta] | None = None,
    inference: str = "auto",
    chunk_size: int | None = None,
) -> None:
    classify_config = config.classify
    if not classify_config.write_full and not classify_config.gwas_ssf_out_file:
        raise new_error(
            "No classification outputs requested. Set classify.write_full=true and/or classify.gwas_ssf_out_file."
        )
    if classify_config.n_samples <= 0:
        classify_config = ClassifyConfig(
            params_override=classify_config.params_override,
            n_steps_burn_in=classify_config.n_steps_burn_in,
            n_samples=min(1000, data.meta.n_data_points()),
            out_file=classify_config.out_file,
            trace_ids=classify_config.trace_ids,
            t_pinned=classify_config.t_pinned,
            write_full=classify_config.write_full,
            gwas_ssf_out_file=classify_config.gwas_ssf_out_file,
            gwas_ssf_guess_fields=classify_config.gwas_ssf_guess_fields,
        )
    if chunk_size is None or chunk_size <= 0:
        chunk_size = _default_chunk_size(data.meta.n_traits(), data.meta.n_data_points())
    use_vectorized = chunk_size > 1 and (
        (inference == "analytic" and not config.outliers.enabled)
        or (inference == "variational" and config.outliers.enabled)
    )
    if use_vectorized:
        classify_vectorized(
            data,
            params,
            classify_config,
            inference,
            chunk_size,
            variant_meta=variant_meta,
        )
        return

    n_threads = max(os.cpu_count() or 1, 3)
    launcher = ClassifyWorkerLauncher(
        data, params, classify_config, inference=inference, outliers_enabled=config.outliers.enabled
    )
    threads = Threads.new(launcher, n_threads)
    meta = data.meta
    out_messages = (MessageToWorker.data_point(i) for i in range(meta.n_data_points()))
    temp_out_file = (
        f"{classify_config.out_file}_tmp" if classify_config.write_full else None
    )
    observer = Observer(meta.var_ids, temp_out_file, meta)
    in_messages = threads.task_queue(out_messages, observer)
    classifications = [message.classification for message in in_messages]
    if classify_config.write_full:
        write_out_file(classify_config.out_file, meta, classifications)
    if classify_config.gwas_ssf_out_file:
        write_gwas_ssf_files(
            classify_config.gwas_ssf_out_file,
            meta,
            classifications,
            variant_meta,
        )
    if temp_out_file and os.path.exists(temp_out_file):
        try:
            os.remove(temp_out_file)
        except OSError as exc:
            print(f"Warning: could not remove temp file {temp_out_file}: {exc}")
    threads.close(MessageToWorker.shutdown())


def classify_vectorized(
    data: GwasData,
    params: Params,
    config: ClassifyConfig,
    inference: str,
    chunk_size: int,
    variant_meta: dict[str, VariantMeta] | None = None,
) -> None:
    if inference not in {"analytic", "variational"}:
        raise new_error(
            "Vectorized classification is only supported for analytic or variational inference."
        )
    meta = data.meta
    n = meta.n_data_points()
    ssf_handles, ssf_header, ssf_include = _maybe_open_ssf_outputs(
        config.gwas_ssf_out_file,
        meta,
        variant_meta,
    )
    full_handle = _open_output(config.out_file) if config.write_full else None
    try:
        if full_handle:
            full_handle.write(format_header(meta))
        if ssf_handles:
            for handle in ssf_handles.values():
                handle.write(ssf_header)
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
                    e_beta_gls, e_se_gls = gls_endophenotype_stats_chunk(
                        params_reduced, single.betas, single.ses
                    )
                    classification = Classification(
                        sampled,
                        mu_calc,
                        e_beta_gls[0],
                        e_se_gls[0],
                    )
                    var_id = single.meta.var_ids[0]
                    if full_handle:
                        full_handle.write(format_entry(var_id, classification))
                    if ssf_handles:
                        _write_ssf_entry(
                            ssf_handles,
                            ssf_include,
                            var_id,
                            classification,
                            meta,
                            variant_meta,
                        )
                continue

            if inference == "analytic":
                sampled = analytical_classification_chunk(params, betas, ses)
                e_beta_gls, e_se_gls = gls_endophenotype_stats_chunk(params, betas, ses)
            elif inference == "variational":
                sampled = outliers_variational_classification_chunk(params, betas, ses)
                e_beta_gls, e_se_gls = outliers_variational_gls_chunk(params, betas, ses)
            else:
                sampled = outliers_analytic_classification_chunk(params, betas, ses)
                e_beta_gls, e_se_gls = outliers_analytic_gls_chunk(params, betas, ses)
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
                    e_beta_gls[i],
                    e_se_gls[i],
                )
                if full_handle:
                    full_handle.write(format_entry(var_id, classification))
                if ssf_handles:
                    _write_ssf_entry(
                        ssf_handles,
                        ssf_include,
                        var_id,
                        classification,
                        meta,
                        variant_meta,
                    )
    finally:
        if full_handle:
            full_handle.close()
        if ssf_handles:
            for handle in ssf_handles.values():
                handle.close()


def write_out_file(file: str, meta: Meta, classifications: list[Classification]) -> None:
    with _open_output(file) as handle:
        handle.write(format_header(meta))
        for var_id, classification in zip(meta.var_ids, classifications):
            handle.write(format_entry(var_id, classification))


def write_gwas_ssf_files(
    out_file: str,
    meta: Meta,
    classifications: list[Classification],
    variant_meta: dict[str, VariantMeta] | None,
) -> None:
    handles, header, include = _maybe_open_ssf_outputs(out_file, meta, variant_meta)
    if not handles:
        return
    try:
        for handle in handles.values():
            handle.write(header)
        for var_id, classification in zip(meta.var_ids, classifications):
            _write_ssf_entry(handles, include, var_id, classification, meta, variant_meta)
    finally:
        for handle in handles.values():
            handle.close()


def _open_output(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return io.open(path, "w", encoding="utf-8")


def format_header(meta: Meta) -> str:
    parts = ["id"]
    for endo in meta.endo_names:
        parts.append(f"{endo}_mean_post")
        parts.append(f"{endo}_std_post")
        parts.append(f"{endo}_mean_calc")
        parts.append(f"{endo}_beta_gls")
        parts.append(f"{endo}_se_gls")
        parts.append(f"{endo}_z_gls")
        parts.append(f"{endo}_p_gls")
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
    e_beta_gls = np.asarray(classification.e_beta_gls, dtype=float)
    e_se_gls = np.asarray(classification.e_se_gls, dtype=float)
    if e_beta_gls.ndim == 2:
        e_beta_gls = e_beta_gls[0]
    if e_se_gls.ndim == 2:
        e_se_gls = e_se_gls[0]
    parts = [var_id]
    for idx in range(len(e_mean)):
        parts.append(str(float(e_mean[idx])))
        parts.append(str(float(e_std[idx])))
        parts.append(str(float(e_calc[idx])))
        beta = float(e_beta_gls[idx]) if idx < len(e_beta_gls) else float("nan")
        se = float(e_se_gls[idx]) if idx < len(e_se_gls) else float("nan")
        if se <= 0.0 or not math.isfinite(beta) or not math.isfinite(se):
            z = float("nan")
            p = float("nan")
        else:
            z = beta / se
            p = math.erfc(abs(z) / math.sqrt(2.0))
        parts.append(str(beta))
        parts.append(str(se))
        parts.append(str(float(z)))
        parts.append(str(float(p)))
    t_means = np.asarray(sampled.t_means, dtype=float)
    if t_means.ndim == 2:
        t_means = t_means[0]
    parts.extend(str(float(value)) for value in t_means)
    return "\t".join(parts) + "\n"


def _maybe_open_ssf_outputs(
    out_file: str | None,
    meta: Meta,
    variant_meta: dict[str, VariantMeta] | None,
) -> tuple[dict[str, io.TextIOBase] | None, str, dict[str, bool]]:
    if not out_file:
        return None, "", {}
    outputs = _ssf_out_paths(out_file, meta.endo_names)
    include = _ssf_meta_columns_present(variant_meta)
    header = _ssf_header(include)
    handles = {endo: _open_output(path) for endo, path in outputs.items()}
    return handles, header, include


def _ssf_out_paths(out_file: str, endo_names: list[str]) -> dict[str, str]:
    if len(endo_names) == 1:
        return {endo_names[0]: out_file}
    gz = out_file.endswith(".gz")
    base = out_file[:-3] if gz else out_file
    stem, ext = os.path.splitext(base)
    outputs = {}
    for endo in endo_names:
        suffix = f"_{endo}"
        if ext:
            path = f"{stem}{suffix}{ext}"
        else:
            path = f"{base}{suffix}"
        if gz:
            path = f"{path}.gz"
        outputs[endo] = path
    return outputs


def _ssf_header(include: dict[str, bool]) -> str:
    parts = ["variant_id"]
    if include["chrom"]:
        parts.append("chromosome")
    if include["pos"]:
        parts.append("base_pair_location")
    if include["effect_allele"]:
        parts.append("effect_allele")
    if include["other_allele"]:
        parts.append("other_allele")
    if include["eaf"]:
        parts.append("effect_allele_frequency")
    if include["rsid"]:
        parts.append("rsid")
    parts.extend(["beta", "standard_error", "p_value"])
    return "\t".join(parts) + "\n"


def _ssf_meta_columns_present(
    variant_meta: dict[str, VariantMeta] | None,
) -> dict[str, bool]:
    fields = {
        "chrom": False,
        "pos": False,
        "effect_allele": False,
        "other_allele": False,
        "eaf": False,
        "rsid": False,
    }
    if not variant_meta:
        return fields
    for meta in variant_meta.values():
        if meta.chrom is not None:
            fields["chrom"] = True
        if meta.pos is not None:
            fields["pos"] = True
        if meta.effect_allele is not None:
            fields["effect_allele"] = True
        if meta.other_allele is not None:
            fields["other_allele"] = True
        if meta.eaf is not None:
            fields["eaf"] = True
        if meta.rsid is not None:
            fields["rsid"] = True
        if all(fields.values()):
            break
    return fields


def _write_ssf_entry(
    handles: dict[str, io.TextIOBase],
    include: dict[str, bool],
    var_id: str,
    classification: Classification,
    meta: Meta,
    variant_meta: dict[str, VariantMeta] | None,
) -> None:
    vmeta = variant_meta.get(var_id) if variant_meta else None
    e_beta = np.asarray(classification.e_beta_gls, dtype=float)
    e_se = np.asarray(classification.e_se_gls, dtype=float)
    if e_beta.ndim == 2:
        e_beta = e_beta[0]
    if e_se.ndim == 2:
        e_se = e_se[0]
    for endo_idx, endo_name in enumerate(meta.endo_names):
        handle = handles[endo_name]
        beta = float(e_beta[endo_idx]) if endo_idx < len(e_beta) else float("nan")
        se = float(e_se[endo_idx]) if endo_idx < len(e_se) else float("nan")
        if se <= 0.0 or not math.isfinite(beta) or not math.isfinite(se):
            p_value = float("nan")
        else:
            z = beta / se
            p_value = math.erfc(abs(z) / math.sqrt(2.0))
        parts = [var_id]
        if include["chrom"]:
            parts.append("" if vmeta is None or vmeta.chrom is None else str(vmeta.chrom))
        if include["pos"]:
            parts.append("" if vmeta is None or vmeta.pos is None else str(vmeta.pos))
        if include["effect_allele"]:
            parts.append(
                "" if vmeta is None or vmeta.effect_allele is None else vmeta.effect_allele
            )
        if include["other_allele"]:
            parts.append(
                "" if vmeta is None or vmeta.other_allele is None else vmeta.other_allele
            )
        if include["eaf"]:
            parts.append("" if vmeta is None or vmeta.eaf is None else str(vmeta.eaf))
        if include["rsid"]:
            parts.append("" if vmeta is None or vmeta.rsid is None else str(vmeta.rsid))
        parts.append(str(beta))
        parts.append(str(se))
        parts.append(str(float(p_value)))
        handle.write("\t".join(parts) + "\n")

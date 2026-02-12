from __future__ import annotations

import gzip
import io
import math
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..check import check_params
from ..data import (
    GwasData,
    LoadedData,
    VariantMeta,
    Weights,
    load_data,
    load_data_bucket,
    estimate_gwas_counts,
    write_flip_log,
    Meta,
)
from ..data.data import FlipStats
from ..data.data import (
    _build_dig_cache,
    _open_text,
    _parse_locus_id,
    _resolve_gwas_path,
)
from ..data.gwas import GwasCols, GwasReader, GwasRecord, default_cols
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


@dataclass
class _Locus:
    chrom: str
    pos: int
    ref: str
    alt: str


@dataclass
class _SortedState:
    trait_name: str
    i_trait: int
    handle: io.TextIOBase
    reader: Iterator[GwasRecord]
    record: GwasRecord | None
    line_no: int = 1
    last_id: str | None = None
    last_chrom: str | None = None
    last_pos: int | None = None
    last_chrom_rank: int = -1


def _sorted_error(message: str) -> Exception:
    return new_error(message + " Rerun with --unsorted to use robust unsorted mode.")


def _record_locus(record: GwasRecord) -> _Locus | None:
    if (
        record.chrom is not None
        and record.pos is not None
        and record.effect_allele is not None
        and record.other_allele is not None
    ):
        chrom = record.chrom
        if chrom.lower().startswith("chr"):
            chrom = chrom[3:]
        return _Locus(
            chrom=chrom,
            pos=record.pos,
            ref=record.other_allele.upper(),
            alt=record.effect_allele.upper(),
        )
    parsed = _parse_locus_id(record.var_id)
    if parsed is None:
        return None
    chrom, pos, ref, alt = parsed
    return _Locus(chrom=chrom, pos=pos, ref=ref, alt=alt)


def _probe_sorted_mode(config: Config, sample_rows: int = 1000) -> str | None:
    if sample_rows <= 0:
        return None
    id_sorted_all = True
    locus_sorted_all = True
    locus_parsed_all = True
    chrom_orders: list[list[str]] = []
    for gwas in config.gwas:
        path = _resolve_gwas_path(gwas, config.data_access)
        cols = gwas.cols or GwasCols(
            id=default_cols.VAR_ID, effect=default_cols.BETA, se=default_cols.SE
        )
        cache = _build_dig_cache(config.data_access)
        with _open_text(
            path,
            retries=config.data_access.retries,
            download=config.data_access.download,
            cache=cache,
        ) as handle:
            reader = GwasReader(handle, cols, auto_meta=False)
            prev_id = None
            seen_chroms: set[str] = set()
            chrom_order: list[str] = []
            last_chrom = None
            last_pos = None
            for _ in range(sample_rows):
                try:
                    rec = next(reader)
                except StopIteration:
                    break
                if prev_id is not None and rec.var_id < prev_id:
                    id_sorted_all = False
                prev_id = rec.var_id
                locus = _record_locus(rec)
                if locus is None:
                    locus_parsed_all = False
                    locus_sorted_all = False
                    continue
                if last_chrom is None:
                    last_chrom = locus.chrom
                    last_pos = locus.pos
                    seen_chroms.add(locus.chrom)
                    chrom_order.append(locus.chrom)
                    continue
                if locus.chrom == last_chrom:
                    if locus.pos < int(last_pos):
                        locus_sorted_all = False
                else:
                    if locus.chrom in seen_chroms:
                        locus_sorted_all = False
                    seen_chroms.add(locus.chrom)
                    chrom_order.append(locus.chrom)
                last_chrom = locus.chrom
                last_pos = locus.pos
            chrom_orders.append(chrom_order)
    if locus_parsed_all and locus_sorted_all and chrom_orders:
        ref = chrom_orders[0]
        ref_idx = {chrom: i for i, chrom in enumerate(ref)}
        order_ok = True
        for order in chrom_orders[1:]:
            idx = [ref_idx[c] for c in order if c in ref_idx]
            if len(idx) != len(order):
                order_ok = False
                break
            if any(idx[i] <= idx[i - 1] for i in range(1, len(idx))):
                order_ok = False
                break
        if order_ok:
            return "locus"
    if id_sorted_all:
        return "id"
    return None


def _resolve_streaming_input_order(config: Config, override: str | None) -> str:
    mode = (override or config.classify.input_order or "auto").lower()
    if mode not in {"auto", "sorted", "unsorted"}:
        raise new_error("classify.input_order must be auto, sorted, or unsorted.")
    if mode == "unsorted":
        return "unsorted"
    probe = _probe_sorted_mode(config, sample_rows=1000)
    if mode == "sorted":
        if probe is None:
            raise _sorted_error(
                "Could not validate sorted ordering from probe rows across GWAS files."
            )
        return probe
    if probe is None:
        return "unsorted"
    return probe


def _sorted_variant_data(
    mode: str,
    records: list[tuple[_SortedState, GwasRecord]],
) -> tuple[str, list[float], list[float], int]:
    n_traits = max(st.i_trait for st, _ in records) + 1
    betas = [float("nan")] * n_traits
    ses = [float("nan")] * n_traits
    if mode == "id":
        out_id = records[0][1].var_id
        for st, rec in records:
            betas[st.i_trait] = rec.beta
            ses[st.i_trait] = rec.se
        observed = sum(
            int(math.isfinite(b) and math.isfinite(s))
            for b, s in zip(betas, ses)
        )
        return out_id, betas, ses, observed
    first_locus = _record_locus(records[0][1])
    if first_locus is None:
        raise _sorted_error(
            f"Variant {records[0][1].var_id} is not locus-parseable in sorted locus mode."
        )
    out_id = f"{first_locus.chrom}:{first_locus.pos}:{first_locus.ref}:{first_locus.alt}"
    for st, rec in records:
        locus = _record_locus(rec)
        if locus is None:
            raise _sorted_error(
                f"Variant {rec.var_id} is not locus-parseable in sorted locus mode."
            )
        beta = rec.beta
        if (
            locus.chrom == first_locus.chrom
            and locus.pos == first_locus.pos
            and locus.ref == first_locus.alt
            and locus.alt == first_locus.ref
        ):
            beta = -beta
        betas[st.i_trait] = beta
        ses[st.i_trait] = rec.se
    observed = sum(int(math.isfinite(b) and math.isfinite(s)) for b, s in zip(betas, ses))
    return out_id, betas, ses, observed


def _sorted_mode_key(
    mode: str,
    state: _SortedState,
    chrom_rank: dict[str, int],
    is_reference: bool,
) -> tuple[tuple[float, float, str, str], tuple[str, int, str, str] | str]:
    rec = state.record
    if rec is None:
        return ((math.inf, math.inf, "", ""), "")
    if mode == "id":
        key = rec.var_id
        if state.last_id is not None and key < state.last_id:
            raise _sorted_error(
                f"GWAS {state.trait_name} is not lexicographically sorted near line {state.line_no}."
            )
        state.last_id = key
        return ((0.0, 0.0, key, ""), key)
    locus = _record_locus(rec)
    if locus is None:
        raise _sorted_error(
            f"GWAS {state.trait_name} has non-locus ID {rec.var_id!r} near line {state.line_no}."
        )
    rank = chrom_rank.get(locus.chrom)
    if is_reference and rank is None:
        rank = len(chrom_rank)
        chrom_rank[locus.chrom] = rank
    if rank is None:
        rank = 10_000_000
    if state.last_chrom is not None:
        if locus.chrom == state.last_chrom and locus.pos < int(state.last_pos):
            raise _sorted_error(
                f"GWAS {state.trait_name} is not sorted within chromosome {locus.chrom} near line {state.line_no}."
            )
        if locus.chrom != state.last_chrom:
            prev_rank = state.last_chrom_rank
            if rank <= prev_rank:
                raise _sorted_error(
                    f"GWAS {state.trait_name} chromosome block order is inconsistent near line {state.line_no}."
                )
    state.last_chrom = locus.chrom
    state.last_pos = locus.pos
    state.last_chrom_rank = rank
    a = locus.ref if locus.ref <= locus.alt else locus.alt
    b = locus.alt if locus.ref <= locus.alt else locus.ref
    return ((float(rank), float(locus.pos), a, b), (locus.chrom, locus.pos, a, b))


def classify_or_check(
    config: Config,
    dry: bool,
    inference: str = "auto",
    chunk_size: int | None = None,
    input_order_override: str | None = None,
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
    if not (0.0 <= config.classify.min_trait_coverage_pct <= 100.0):
        raise new_error("classify.min_trait_coverage_pct must be between 0 and 100.")
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
    max_memory_gb = config.data_access.max_memory_gb
    use_streaming = False
    n_variants_est = None
    if max_memory_gb and max_memory_gb > 0:
        counts = estimate_gwas_counts(config)
        if counts:
            n_variants_est = max(counts.values())
            est_bytes = _estimate_classify_memory_bytes(
                n_variants_est, len(config.gwas)
            )
            max_bytes = max_memory_gb * (1024**3)
            use_streaming = est_bytes > max_bytes
            print(
                "Estimated classification memory: {:.2f} GB (max {:.2f} GB).".format(
                    est_bytes / (1024**3), max_memory_gb
                )
            )
    if use_streaming:
        if inference_mode == "gibbs":
            raise new_error(
                "Streaming classification does not support Gibbs sampling. "
                "Increase data_access.max_memory_gb or use analytic/variational inference."
            )
        if dry:
            print("User picked dry run only, so doing nothing.")
            return
        classify_streaming(
            config,
            params,
            inference=inference_mode,
            chunk_size=chunk_size,
            n_variants_est=n_variants_est,
            input_order_override=input_order_override,
        )
        return
    data = load_data(config, Action.CLASSIFY)
    data = _filter_data_by_trait_coverage(data, config.classify.min_trait_coverage_pct)
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


def _estimate_classify_memory_bytes(n_variants: int, n_traits: int) -> int:
    bytes_arrays = n_variants * n_traits * 2 * 8
    bytes_overhead = n_variants * 1024
    return int(bytes_arrays + bytes_overhead)


def classify_streaming(
    config: Config,
    params: Params,
    inference: str,
    chunk_size: int | None,
    n_variants_est: int | None,
    input_order_override: str | None = None,
) -> None:
    classify_config = config.classify
    if not classify_config.write_full and not classify_config.gwas_ssf_out_file:
        raise new_error(
            "No classification outputs requested. Set classify.write_full=true and/or classify.gwas_ssf_out_file."
        )
    if inference not in {"analytic", "variational"}:
        raise new_error(
            "Streaming classification is only supported for analytic or variational inference."
        )
    if n_variants_est is None:
        counts = estimate_gwas_counts(config)
        n_variants_est = max(counts.values()) if counts else 0
    stream_mode = _resolve_streaming_input_order(config, input_order_override)
    if stream_mode in {"id", "locus"}:
        print(f"Using sorted single-pass streaming mode ({stream_mode}).")
        _classify_streaming_sorted(
            config,
            params,
            inference,
            chunk_size=chunk_size,
            mode=stream_mode,
        )
        return

    est_bytes = _estimate_classify_memory_bytes(n_variants_est, len(config.gwas))
    max_bytes = config.data_access.max_memory_gb * (1024**3)
    n_buckets = max(1, math.ceil(est_bytes / max_bytes)) if max_bytes > 0 else 1
    print(f"Using unsorted streaming classification with {n_buckets} buckets.")

    full_handle = _open_output(classify_config.out_file) if classify_config.write_full else None
    ssf_handles = None
    ssf_header = ""
    ssf_include: dict[str, bool] = {}
    flip_stats: dict[str, FlipStats] = {}
    try:
        for bucket in range(n_buckets):
            data = load_data_bucket(
                config,
                Action.CLASSIFY,
                bucket,
                n_buckets,
                flip_stats=flip_stats,
            )
            data = _filter_data_by_trait_coverage(
                data,
                classify_config.min_trait_coverage_pct,
            )
            if data.gwas_data.meta.n_data_points() == 0:
                continue
            if chunk_size is None or chunk_size <= 0:
                chunk_size = _default_chunk_size(
                    data.gwas_data.meta.n_traits(), data.gwas_data.meta.n_data_points()
                )
            write_header = bucket == 0
            if ssf_handles is None and classify_config.gwas_ssf_out_file:
                ssf_handles, ssf_header, ssf_include = _maybe_open_ssf_outputs(
                    classify_config.gwas_ssf_out_file,
                    data.gwas_data.meta,
                    data.variant_meta,
                )
            _classify_vectorized_write(
                data.gwas_data,
                params,
                classify_config,
                inference,
                chunk_size,
                data.variant_meta,
                full_handle,
                ssf_handles,
                ssf_header,
                ssf_include,
                write_header,
                False,
            )
    finally:
        if full_handle:
            full_handle.close()
        if ssf_handles:
            for handle in ssf_handles.values():
                handle.close()
    write_flip_log(config.files.flip_log, flip_stats)


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


def _classify_streaming_sorted(
    config: Config,
    params: Params,
    inference: str,
    chunk_size: int | None,
    mode: str,
) -> None:
    classify_config = config.classify
    trait_names = [g.name for g in config.gwas]
    endo_names = list(params.endo_names)
    n_traits = len(trait_names)
    min_required = int(math.ceil((classify_config.min_trait_coverage_pct / 100.0) * max(1, n_traits)))
    if chunk_size is None or chunk_size <= 0:
        chunk_size = _default_chunk_size(n_traits, 0)

    full_handle = _open_output(classify_config.out_file) if classify_config.write_full else None
    ssf_handles, ssf_header, ssf_include = _maybe_open_ssf_outputs(
        classify_config.gwas_ssf_out_file,
        Meta(trait_names=trait_names, var_ids=[], endo_names=endo_names),
        None,
    )
    write_header = True
    flip_stats = {g.name: FlipStats() for g in config.gwas}
    states: list[_SortedState] = []
    handles: list[io.TextIOBase] = []
    chrom_rank: dict[str, int] = {}
    try:
        for i_trait, g in enumerate(config.gwas):
            cols = g.cols or GwasCols(
                id=default_cols.VAR_ID,
                effect=default_cols.BETA,
                se=default_cols.SE,
            )
            path = _resolve_gwas_path(g, config.data_access)
            cache = _build_dig_cache(config.data_access)
            handle = _open_text(
                path,
                retries=config.data_access.retries,
                download=config.data_access.download,
                cache=cache,
            )
            handles.append(handle)
            reader = GwasReader(handle, cols, auto_meta=False)
            try:
                first = next(reader)
            except StopIteration:
                first = None
            states.append(
                _SortedState(
                    trait_name=g.name,
                    i_trait=i_trait,
                    handle=handle,
                    reader=reader,
                    record=first,
                )
            )

        chunk_ids: list[str] = []
        chunk_betas: list[list[float]] = []
        chunk_ses: list[list[float]] = []
        while True:
            active = [st for st in states if st.record is not None]
            if not active:
                break
            keyed: list[tuple[_SortedState, tuple[float, float, str, str], tuple[str, int, str, str] | str]] = []
            for st in active:
                sort_key, match_key = _sorted_mode_key(
                    mode,
                    st,
                    chrom_rank=chrom_rank,
                    is_reference=(st.i_trait == 0),
                )
                keyed.append((st, sort_key, match_key))
            keyed.sort(key=lambda x: x[1])
            target_match = keyed[0][2]
            matched: list[tuple[_SortedState, GwasRecord]] = []
            for st, _sort_key, match_key in keyed:
                if match_key == target_match and st.record is not None:
                    matched.append((st, st.record))
            out_id, row_betas, row_ses, observed = _sorted_variant_data(mode, matched)
            for st, rec in matched:
                stats = flip_stats[st.trait_name]
                stats.matched += 1
                if mode == "locus":
                    first = _record_locus(matched[0][1])
                    loc = _record_locus(rec)
                    if first is not None and loc is not None:
                        if (
                            loc.chrom == first.chrom
                            and loc.pos == first.pos
                            and loc.ref == first.alt
                            and loc.alt == first.ref
                        ):
                            stats.flipped += 1
                st.line_no += 1
                try:
                    st.record = next(st.reader)
                except StopIteration:
                    st.record = None
            if observed >= min_required:
                chunk_ids.append(out_id)
                chunk_betas.append(row_betas)
                chunk_ses.append(row_ses)
            if len(chunk_ids) >= chunk_size:
                data = GwasData(
                    meta=Meta(trait_names=trait_names, var_ids=list(chunk_ids), endo_names=endo_names),
                    betas=np.asarray(chunk_betas, dtype=float),
                    ses=np.asarray(chunk_ses, dtype=float),
                )
                _classify_vectorized_write(
                    data,
                    params,
                    classify_config,
                    inference,
                    chunk_size,
                    None,
                    full_handle,
                    ssf_handles,
                    ssf_header,
                    ssf_include,
                    write_header,
                    False,
                )
                write_header = False
                chunk_ids.clear()
                chunk_betas.clear()
                chunk_ses.clear()
        if chunk_ids:
            data = GwasData(
                meta=Meta(trait_names=trait_names, var_ids=list(chunk_ids), endo_names=endo_names),
                betas=np.asarray(chunk_betas, dtype=float),
                ses=np.asarray(chunk_ses, dtype=float),
            )
            _classify_vectorized_write(
                data,
                params,
                classify_config,
                inference,
                chunk_size,
                None,
                full_handle,
                ssf_handles,
                ssf_header,
                ssf_include,
                write_header,
                False,
            )
    finally:
        for handle in handles:
            try:
                handle.close()
            except OSError:
                pass
        if full_handle:
            full_handle.close()
        if ssf_handles:
            for handle in ssf_handles.values():
                handle.close()
    write_flip_log(config.files.flip_log, flip_stats)


def _classify_vectorized_write(
    data: GwasData,
    params: Params,
    config: ClassifyConfig,
    inference: str,
    chunk_size: int,
    variant_meta: dict[str, VariantMeta] | None,
    full_handle: io.TextIOBase | None,
    ssf_handles: dict[str, io.TextIOBase] | None,
    ssf_header: str,
    ssf_include: dict[str, bool],
    write_header: bool,
    close_handles: bool,
) -> None:
    meta = data.meta
    n = meta.n_data_points()
    try:
        if full_handle and write_header:
            full_handle.write(format_header(meta))
        if ssf_handles and write_header:
            for handle in ssf_handles.values():
                handle.write(ssf_header)
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            betas = data.betas[start:end, :]
            ses = data.ses[start:end, :]
            n_traits_observed_chunk = np.sum(
                np.isfinite(betas) & np.isfinite(ses),
                axis=1,
            )
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
                        len(is_col),
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
                    int(n_traits_observed_chunk[i]),
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
        if close_handles:
            if full_handle:
                full_handle.close()
            if ssf_handles:
                for handle in ssf_handles.values():
                    handle.close()


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
    ssf_handles, ssf_header, ssf_include = _maybe_open_ssf_outputs(
        config.gwas_ssf_out_file,
        meta,
        variant_meta,
    )
    full_handle = _open_output(config.out_file) if config.write_full else None
    _classify_vectorized_write(
        data,
        params,
        config,
        inference,
        chunk_size,
        variant_meta,
        full_handle,
        ssf_handles,
        ssf_header,
        ssf_include,
        True,
        True,
    )


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


def _open_output(path: str, mode: str = "w"):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return io.open(path, mode, encoding="utf-8")


def format_header(meta: Meta) -> str:
    parts = ["id", "n_traits_observed"]
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
    parts = [var_id, str(int(classification.n_traits_observed))]
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


def _filter_data_by_trait_coverage(
    loaded: LoadedData,
    min_coverage_pct: float,
) -> LoadedData:
    if min_coverage_pct <= 0.0:
        return loaded
    data = loaded.gwas_data
    n_traits = max(1, data.meta.n_traits())
    required = int(math.ceil((min_coverage_pct / 100.0) * n_traits))
    if required <= 0:
        return loaded
    observed = np.sum(np.isfinite(data.betas) & np.isfinite(data.ses), axis=1)
    keep = observed >= required
    if np.all(keep):
        return loaded
    kept = int(np.sum(keep))
    total = int(data.meta.n_data_points())
    print(
        "Filtered variants by trait coverage: kept {} of {} "
        "(min_trait_coverage_pct={} => >= {} / {} traits).".format(
            kept,
            total,
            min_coverage_pct,
            required,
            n_traits,
        )
    )
    indices = np.nonzero(keep)[0]
    var_ids = [data.meta.var_ids[i] for i in indices]
    gwas_data = GwasData(
        meta=Meta(
            trait_names=data.meta.trait_names,
            var_ids=var_ids,
            endo_names=data.meta.endo_names,
        ),
        betas=data.betas[indices, :],
        ses=data.ses[indices, :],
    )
    variant_meta = None
    if loaded.variant_meta is not None:
        variant_meta = {
            var_id: loaded.variant_meta[var_id]
            for var_id in var_ids
            if var_id in loaded.variant_meta
        }
    weights = loaded.weights
    filtered_weights = Weights(
        weights=[weights.weights[i] for i in indices if i < len(weights.weights)],
        sum=float(
            sum(weights.weights[i] for i in indices if i < len(weights.weights))
        ),
    )
    return LoadedData(
        gwas_data=gwas_data,
        weights=filtered_weights,
        variant_meta=variant_meta,
    )


def _maybe_open_ssf_outputs(
    out_file: str | None,
    meta: Meta,
    variant_meta: dict[str, VariantMeta] | None,
    mode: str = "w",
) -> tuple[dict[str, io.TextIOBase] | None, str, dict[str, bool]]:
    if not out_file:
        return None, "", {}
    outputs = _ssf_out_paths(out_file, meta.endo_names)
    include = _ssf_meta_columns_present(variant_meta)
    header = _ssf_header(include)
    handles = {endo: _open_output(path, mode=mode) for endo, path in outputs.items()}
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

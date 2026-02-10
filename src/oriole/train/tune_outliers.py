from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math
import random
import time

import numpy as np

from ..data import GwasData, load_data_for_ids, subset_gwas_data, read_id_keys
from ..data.data import resolve_variant_mode
from ..options.config import Config
from ..options.action import Action
from ..options.inference import resolve_inference
from ..params import Params
from ..train.initial_params import estimate_initial_params
from ..train.analytical import estimate_params_analytical
from ..train.outliers_analytic import estimate_params_analytical_outliers
from ..train.outliers_variational import estimate_params_variational_outliers
from ..classify.analytical import analytical_classification_chunk
from ..classify.outliers_analytic import outliers_analytic_classification_chunk
from ..classify.outliers_variational import outliers_variational_classification_chunk
from ..sample.sampler import Sampler
from ..sample.vars import Vars
from ..options.config import endophenotype_mask


@dataclass
class TuneResult:
    kappa: float
    expected_outliers: float
    pi: float
    score: float
    scores: dict[tuple[float, float], float]
    kappa_grid: list[float]
    expected_outliers_grid: list[float]


@dataclass
class TuneCache:
    positives: list[str]
    folds: list[list[str]]
    background_ids: list[str]
    hard_ids: list[str]
    pos_data: GwasData
    background_data: GwasData | None
    hard_data: GwasData | None
    pos_index: dict[str, int]


def _read_ids(ids_file: str, variant_mode: str) -> list[str]:
    return read_id_keys(ids_file, variant_mode)


def _sample_ids(ids: list[str], n: int, rng: random.Random) -> list[str]:
    if n >= len(ids):
        return list(ids)
    return rng.sample(ids, n)


def _folds(ids: list[str], n_folds: int, rng: random.Random) -> list[list[str]]:
    shuffled = list(ids)
    rng.shuffle(shuffled)
    folds = [shuffled[i::n_folds] for i in range(n_folds)]
    return folds


def _score_from_posterior(e_mean: np.ndarray, e_std: np.ndarray) -> float:
    denom = np.maximum(e_std, 1e-12)
    return float(np.sum((e_mean / denom) ** 2))


def _scores_for_data(
    data: GwasData,
    params: Params,
    inference: str,
    n_burn_in: int,
    n_samples: int,
) -> list[float]:
    if inference == "analytic":
        sampled = outliers_analytic_classification_chunk(params, data.betas, data.ses)
        return [
            _score_from_posterior(sampled.e_mean[i], sampled.e_std[i])
            for i in range(data.meta.n_data_points())
        ]
    if inference == "variational":
        sampled = outliers_variational_classification_chunk(params, data.betas, data.ses)
        return [
            _score_from_posterior(sampled.e_mean[i], sampled.e_std[i])
            for i in range(data.meta.n_data_points())
        ]
    if inference == "gibbs":
        rng = np.random.default_rng()
        scores: list[float] = []
        for i in range(data.meta.n_data_points()):
            single, _ = data.only_data_point(i)
            vars = Vars.initial_vars(single, params)
            sampler = Sampler(single.meta, rng)
            sampler.sample_n(single, params, vars, n_burn_in, None, False)
            sampler.reset_stats()
            sampler.sample_n(single, params, vars, n_samples, None, False)
            sampled = sampler.var_stats.calculate_classification()
            scores.append(_score_from_posterior(sampled.e_mean[0], sampled.e_std[0]))
        return scores
    # fallback for base model
    sampled = analytical_classification_chunk(params, data.betas, data.ses)
    return [
        _score_from_posterior(sampled.e_mean[i], sampled.e_std[i])
        for i in range(data.meta.n_data_points())
    ]


def _threshold_for_fpr(scores: list[float], alpha: float) -> float:
    if not scores:
        return float("inf")
    n = len(scores)
    k = max(1, int(math.ceil(alpha * n)))
    scores_sorted = sorted(scores, reverse=True)
    return scores_sorted[k - 1]


def _metric(
    scores_pos: list[float],
    scores_bg: list[float],
    scores_hard: list[float] | None,
    fpr_targets: list[float],
    lambda_hard: float,
) -> float:
    if not scores_pos or not scores_bg:
        return float("-inf")
    hard = scores_hard or []
    out = []
    for alpha in fpr_targets:
        threshold = _threshold_for_fpr(scores_bg, alpha)
        tpr = sum(1 for s in scores_pos if s >= threshold) / len(scores_pos)
        if hard:
            fpr_hard = sum(1 for s in hard if s >= threshold) / len(hard)
        else:
            fpr_hard = 0.0
        out.append(tpr - lambda_hard * fpr_hard)
    return float(sum(out) / len(out))


def _train_params_for_ids(
    config: Config,
    ids: list[str],
    inference: str,
    kappa: float,
    expected_outliers: float,
    pos_data: GwasData | None = None,
    pos_index: dict[str, int] | None = None,
) -> Params:
    if pos_data is not None and pos_index is not None:
        try:
            indices = [pos_index[var_id] for var_id in ids]
        except KeyError as exc:
            raise ValueError(f"Missing positive ID in cached data: {exc.args[0]}") from exc
        data = subset_gwas_data(pos_data, indices)
    else:
        data = load_data_for_ids(config, ids)
    mask = np.asarray(endophenotype_mask(config), dtype=bool)
    trait_names = [item.name for item in config.gwas]
    n_traits = len(trait_names)
    trait_index = {name: idx for idx, name in enumerate(trait_names)}
    parent_mask = np.zeros((len(trait_names), len(trait_names)), dtype=bool)
    for edge in config.trait_edges:
        parent_mask[trait_index[edge.child], trait_index[edge.parent]] = True
    params = estimate_initial_params(data, config.endophenotypes, mask, match_rust=False)
    if not config.train.learn_mu_tau:
        params.mus = [config.train.mu for _ in params.endo_names]
        params.taus = [config.train.tau for _ in params.endo_names]
    params.outlier_kappa = kappa
    pi = expected_outliers / max(1, n_traits)
    params.outlier_pis = [pi for _ in params.trait_names]
    chunk_size = max(1, data.meta.n_data_points())
    if inference == "analytic":
        params = estimate_params_analytical_outliers(
            LoadedDataShim(data), params, chunk_size, mask, parent_mask
        )
        if not config.train.learn_mu_tau:
            params.mus = [config.train.mu for _ in params.endo_names]
            params.taus = [config.train.tau for _ in params.endo_names]
        return params
    if inference == "variational":
        params = estimate_params_variational_outliers(
            LoadedDataShim(data), params, chunk_size, mask, parent_mask
        )
        if not config.train.learn_mu_tau:
            params.mus = [config.train.mu for _ in params.endo_names]
            params.taus = [config.train.tau for _ in params.endo_names]
        return params
    raise ValueError("Full tuning with gibbs inference is not supported.")


@dataclass
class LoadedDataShim:
    gwas_data: GwasData
    weights: list[float] | None = None

    def __init__(self, data: GwasData) -> None:
        self.gwas_data = data
        self.weights = type(
            "WeightsShim", (), {"weights": [1.0] * data.meta.n_data_points(), "sum": data.meta.n_data_points()}
        )()


def build_tune_cache(config: Config) -> TuneCache:
    tune = config.tune_outliers
    rng = random.Random(tune.seed)
    positives = _read_ids(config.train.ids_file, resolve_variant_mode(config, Action.TRAIN))
    folds = _folds(positives, tune.n_folds, rng)

    background_ids: list[str] = []
    genome_ids: list[str] = []
    if tune.genomewide_ids_file:
        genome_ids = _read_ids(
            tune.genomewide_ids_file, resolve_variant_mode(config, Action.TRAIN)
        )
        genome_ids = [v for v in genome_ids if v not in set(positives)]
        if tune.n_background_sample > len(genome_ids):
            raise ValueError(
                "tune_outliers.n_background_sample exceeds available genomewide IDs "
                f"({tune.n_background_sample} requested, {len(genome_ids)} available)."
            )
        background_ids = _sample_ids(genome_ids, tune.n_background_sample, rng)

    hard_ids: list[str] = []
    if tune.negative_ids_file:
        neg_ids = _read_ids(
            tune.negative_ids_file, resolve_variant_mode(config, Action.TRAIN)
        )
        neg_ids = [v for v in neg_ids if v not in set(positives)]
        if tune.n_negative_sample > len(neg_ids):
            raise ValueError(
                "tune_outliers.n_negative_sample exceeds available negative IDs "
                f"({tune.n_negative_sample} requested, {len(neg_ids)} available)."
            )
        hard_ids = _sample_ids(neg_ids, tune.n_negative_sample, rng)
        if not hard_ids:
            print(
                "Warning: tune_outliers.negative_ids_file produced no usable IDs; "
                "hard-negative term will be skipped."
            )

    combined_ids: list[str] = []
    seen: set[str] = set()
    for group in (positives, background_ids, hard_ids):
        for var_id in group:
            if var_id not in seen:
                seen.add(var_id)
                combined_ids.append(var_id)

    print(f"Loading combined IDs once (n={len(combined_ids)})")
    load_start = time.perf_counter()
    combined = load_data_for_ids(config, combined_ids)
    print(f"Loaded combined data in {time.perf_counter() - load_start:.2f}s")
    combined_index = {var_id: idx for idx, var_id in enumerate(combined.meta.var_ids)}

    def _subset(ids: list[str]) -> GwasData | None:
        if not ids:
            return None
        try:
            indices = [combined_index[var_id] for var_id in ids]
        except KeyError as exc:
            raise ValueError(f"Missing ID in combined data: {exc.args[0]}") from exc
        return subset_gwas_data(combined, indices)

    pos_data = _subset(positives)
    if pos_data is None:
        raise ValueError("No positive IDs available for tuning.")

    return TuneCache(
        positives=positives,
        folds=folds,
        background_ids=background_ids,
        hard_ids=hard_ids,
        pos_data=pos_data,
        background_data=_subset(background_ids),
        hard_data=_subset(hard_ids),
        pos_index={var_id: idx for idx, var_id in enumerate(pos_data.meta.var_ids)},
    )


def tune_outliers(
    config: Config,
    inference: str,
    cache: TuneCache | None = None,
) -> TuneResult:
    tune = config.tune_outliers
    if not config.outliers.enabled:
        raise ValueError("Outlier tuning requires outliers.enabled = true.")
    overall_start = time.perf_counter()
    if cache is None:
        cache = build_tune_cache(config)
    positives = cache.positives
    folds = cache.folds
    pos_data = cache.pos_data
    pos_index = cache.pos_index

    background_ids = cache.background_ids
    background_data = cache.background_data
    hard_ids = cache.hard_ids
    hard_data = cache.hard_data

    if tune.mode == "auto":
        if config.trait_edges or inference != "analytic":
            mode = "fast"
        else:
            mode = "full"
    else:
        mode = tune.mode

    base_params: Params | None = None
    if mode == "fast":
        base_data = load_data_for_ids(config, positives)
        mask = np.asarray(endophenotype_mask(config), dtype=bool)
        trait_names = [item.name for item in config.gwas]
        trait_index = {name: idx for idx, name in enumerate(trait_names)}
        parent_mask = np.zeros((len(trait_names), len(trait_names)), dtype=bool)
        for edge in config.trait_edges:
            parent_mask[trait_index[edge.child], trait_index[edge.parent]] = True
        base_params = estimate_initial_params(base_data, config.endophenotypes, mask)
        if not config.train.learn_mu_tau:
            base_params.mus = [config.train.mu for _ in base_params.endo_names]
            base_params.taus = [config.train.tau for _ in base_params.endo_names]
        base_params.outlier_kappa = config.outliers.kappa
        base_params.outlier_pis = [config.outliers.pi for _ in base_params.trait_names]
        chunk_size = max(1, base_data.meta.n_data_points())
        if inference == "analytic":
            base_params = estimate_params_analytical_outliers(
                LoadedDataShim(base_data), base_params, chunk_size, mask, parent_mask
            )
        elif inference == "variational":
            base_params = estimate_params_variational_outliers(
                LoadedDataShim(base_data), base_params, chunk_size, mask, parent_mask
            )
        else:
            base_params = estimate_params_variational_outliers(
                LoadedDataShim(base_data), base_params, chunk_size, mask, parent_mask
            )
        if not config.train.learn_mu_tau:
            base_params.mus = [config.train.mu for _ in base_params.endo_names]
            base_params.taus = [config.train.tau for _ in base_params.endo_names]

    scores: dict[tuple[float, float], float] = {}
    fold_scores: dict[tuple[float, float], list[float]] = {}

    def _best_from_scores(
        kappa_grid: list[float],
        expected_grid: list[float],
    ) -> TuneResult:
        means = {
            key: float(np.mean(values))
            for key, values in fold_scores.items()
            if values
        }
        if not means:
            chosen_kappa = kappa_grid[0]
            chosen_expected = expected_grid[0]
            return TuneResult(
                kappa=chosen_kappa,
                expected_outliers=chosen_expected,
                pi=chosen_expected / max(1, len(config.gwas)),
                score=float("-inf"),
                scores={},
                kappa_grid=list(kappa_grid),
                expected_outliers_grid=list(expected_grid),
            )
        best_key, best_mean = max(means.items(), key=lambda item: item[1])
        best_values = fold_scores[best_key]
        best_std = float(np.std(best_values, ddof=1)) if len(best_values) > 1 else 0.0
        best_se = best_std / math.sqrt(max(1, len(best_values)))
        threshold = best_mean - best_se
        candidates = [
            (kappa, expected)
            for (kappa, expected), mean in means.items()
            if mean >= threshold
        ]
        candidates.sort(key=lambda item: (item[1], item[0]))
        chosen_kappa, chosen_expected = candidates[0]
        return TuneResult(
            kappa=chosen_kappa,
            expected_outliers=chosen_expected,
            pi=chosen_expected / max(1, len(config.gwas)),
            score=means[(chosen_kappa, chosen_expected)],
            scores=means,
            kappa_grid=list(kappa_grid),
            expected_outliers_grid=list(expected_grid),
        )

    def evaluate(
        kappa_grid: list[float],
        expected_grid: list[float],
        points: list[tuple[float, float]] | None = None,
    ) -> TuneResult:
        if points is None:
            points = [
                (kappa, expected) for kappa in kappa_grid for expected in expected_grid
            ]
        grid_start = time.perf_counter()
        n_traits = len(config.gwas)
        for kappa, expected in points:
            pi = expected / max(1, n_traits)
            point_start = time.perf_counter()
            print(
                f"CV grid point kappa={kappa} expected_outliers={expected} (pi={pi:.6g}) start"
            )
            if (kappa, expected) in scores:
                score = scores[(kappa, expected)]
            else:
                fold_scores_point = []
                for i_fold, fold in enumerate(folds, start=1):
                    fold_start = time.perf_counter()
                    print(f"  Fold {i_fold}/{len(folds)} start")
                    train_ids = [v for v in positives if v not in fold]
                    params = base_params
                    if mode == "full":
                        train_start = time.perf_counter()
                        params = _train_params_for_ids(
                            config,
                            train_ids,
                            inference,
                            kappa,
                            expected,
                            pos_data=pos_data,
                            pos_index=pos_index,
                        )
                        print(
                            f"    Training for fold {i_fold} "
                            f"took {time.perf_counter() - train_start:.2f}s"
                        )
                    if params is None:
                        raise ValueError("Missing params for tuning.")
                    params = Params(
                        trait_names=params.trait_names,
                        endo_names=params.endo_names,
                        mus=params.mus,
                        taus=params.taus,
                        betas=params.betas,
                        sigmas=params.sigmas,
                        trait_edges=params.trait_edges,
                        outlier_kappa=kappa,
                        outlier_pis=[pi for _ in params.trait_names],
                    )
                    try:
                        val_indices = [pos_index[var_id] for var_id in fold]
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing positive ID in cached data: {exc.args[0]}"
                        ) from exc
                    data_val = subset_gwas_data(pos_data, val_indices)
                    score_start = time.perf_counter()
                    scores_pos = _scores_for_data(
                        data_val,
                        params,
                        inference,
                        config.train.n_steps_burn_in,
                        config.train.n_samples_per_iteration,
                    )
                    scores_bg = []
                    if background_data is not None:
                        scores_bg = _scores_for_data(
                            background_data,
                            params,
                            inference,
                            config.train.n_steps_burn_in,
                            config.train.n_samples_per_iteration,
                        )
                    scores_hard = []
                    if hard_data is not None:
                        scores_hard = _scores_for_data(
                            hard_data,
                            params,
                            inference,
                            config.train.n_steps_burn_in,
                            config.train.n_samples_per_iteration,
                        )
                    print(
                        f"    Scoring for fold {i_fold} "
                        f"took {time.perf_counter() - score_start:.2f}s"
                    )
                    fold_scores_point.append(
                        _metric(
                            scores_pos,
                            scores_bg,
                            scores_hard if hard_ids else None,
                            tune.fpr_targets,
                            tune.lambda_hard if hard_ids else 0.0,
                        )
                    )
                    print(f"  Fold {i_fold} done in {time.perf_counter() - fold_start:.2f}s")
                score = float(np.mean(fold_scores_point))
                scores[(kappa, expected)] = score
                fold_scores[(kappa, expected)] = fold_scores_point
            print(
                f"CV grid point kappa={kappa} expected_outliers={expected} done in "
                f"{time.perf_counter() - point_start:.2f}s score={score:.4f}"
            )
        print(f"Completed CV grid in {time.perf_counter() - grid_start:.2f}s")
        return _best_from_scores(kappa_grid, expected_grid)

    def _expand_to_min(
        values: list[float],
        min_size: int,
        max_size: int,
        lower_bound: float,
        upper_bound: float,
    ) -> list[float]:
        expanded = list(sorted(set(values)))
        while len(expanded) < min_size and len(expanded) < max_size:
            added = []
            lower = max(lower_bound, expanded[0] / tune.expansion_factor)
            upper = min(upper_bound, expanded[-1] * tune.expansion_factor)
            if lower < expanded[0]:
                added.append(lower)
            if upper > expanded[-1]:
                added.append(upper)
            if not added:
                break
            expanded = sorted(set(expanded + added))
        return expanded

    kappa_grid = sorted(set(tune.kappa_grid))
    expected_grid = sorted(set(tune.expected_outliers_grid))
    if not tune.kappa_grid_specified:
        kappa_grid = [config.outliers.kappa]
    if not tune.expected_outliers_grid_specified:
        if config.outliers.expected_outliers is not None:
            expected_grid = [config.outliers.expected_outliers]
        else:
            expected_grid = [config.outliers.pi * max(1, len(config.gwas))]
    if tune.min_grid_length > 1:
        kappa_grid = _expand_to_min(
            kappa_grid,
            tune.min_grid_length,
            tune.max_grid_length,
            1.0,
            float("inf"),
        )
        expected_grid = _expand_to_min(
            expected_grid,
            tune.min_grid_length,
            tune.max_grid_length,
            1e-6,
            max(1e-6, len(config.gwas) - 1e-6),
        )
    best = evaluate(kappa_grid, expected_grid)

    expansions = 0
    max_expansions = tune.max_expansions
    max_grid_size = max(1, tune.max_grid_length)
    def _best_score() -> float:
        if not scores:
            return float("-inf")
        return max(scores.values())

    while tune.expand_on_boundary:
        if max_expansions > 0 and expansions >= max_expansions:
            print(
                "Stopping grid expansion after reaching max_expansions="
                f"{max_expansions}."
            )
            break
        best_score = _best_score()
        threshold = best_score * (1.0 - max(0.0, tune.boundary_margin))
        top_row = expected_grid[-1]
        bottom_row = expected_grid[0]
        left_col = kappa_grid[0]
        right_col = kappa_grid[-1]
        row_top_hit = any(
            scores.get((kappa, top_row), float("-inf")) >= threshold
            for kappa in kappa_grid
        )
        row_bottom_hit = any(
            scores.get((kappa, bottom_row), float("-inf")) >= threshold
            for kappa in kappa_grid
        )
        col_left_hit = any(
            scores.get((left_col, expected), float("-inf")) >= threshold
            for expected in expected_grid
        )
        col_right_hit = any(
            scores.get((right_col, expected), float("-inf")) >= threshold
            for expected in expected_grid
        )
        if not tune.force_expansions:
            if not (row_top_hit or row_bottom_hit or col_left_hit or col_right_hit):
                break
        if len(kappa_grid) >= max_grid_size and len(expected_grid) >= max_grid_size:
            print(
                "Boundary optimum detected but max_grid_size reached "
                f"(kappa={len(kappa_grid)}, expected_outliers={len(expected_grid)}; "
                f"max_grid_size={max_grid_size})."
            )
            break
        new_kappa_values: list[float] = []
        new_expected_values: list[float] = []
        if tune.force_expansions:
            if len(kappa_grid) < max_grid_size:
                new_kappa_values.append(kappa_grid[-1] * tune.expansion_factor)
            if len(expected_grid) < max_grid_size:
                candidate = min(
                    len(config.gwas) - 1e-6, expected_grid[-1] * tune.expansion_factor
                )
                if candidate > expected_grid[-1]:
                    new_expected_values.append(candidate)
        else:
            if row_bottom_hit and len(expected_grid) < max_grid_size:
                candidate = max(1e-6, expected_grid[0] / tune.expansion_factor)
                if candidate < expected_grid[0]:
                    new_expected_values.append(candidate)
            if row_top_hit and len(expected_grid) < max_grid_size:
                candidate = min(
                    len(config.gwas) - 1e-6, expected_grid[-1] * tune.expansion_factor
                )
                if candidate > expected_grid[-1]:
                    new_expected_values.append(candidate)
            if col_left_hit and len(kappa_grid) < max_grid_size:
                candidate = max(1.0, kappa_grid[0] / tune.expansion_factor)
                if candidate < kappa_grid[0]:
                    new_kappa_values.append(candidate)
            if col_right_hit and len(kappa_grid) < max_grid_size:
                new_kappa_values.append(kappa_grid[-1] * tune.expansion_factor)
        new_kappa_values = sorted(set(new_kappa_values))
        new_expected_values = sorted(set(new_expected_values))
        if not new_kappa_values and not new_expected_values:
            print(
                "Boundary optimum detected but no new grid points could be added "
                f"(kappa={len(kappa_grid)}, expected_outliers={len(expected_grid)}; "
                f"max_grid_size={max_grid_size})."
            )
            break
        kappa_grid = sorted(set(kappa_grid + new_kappa_values))
        expected_grid = sorted(set(expected_grid + new_expected_values))
        expansions += 1
        points_to_eval: list[tuple[float, float]] = []
        for kappa in new_kappa_values:
            for expected in expected_grid:
                points_to_eval.append((kappa, expected))
        for expected in new_expected_values:
            for kappa in kappa_grid:
                points_to_eval.append((kappa, expected))
        print(
            "Boundary optimum detected; expanding grid with new points "
            f"kappa={new_kappa_values or '[]'} expected_outliers={new_expected_values or '[]'} "
            f"(expansion {expansions})."
        )
        best = evaluate(kappa_grid, expected_grid, points=points_to_eval)

    if (
        best.kappa == kappa_grid[0]
        or best.kappa == kappa_grid[-1]
        or best.expected_outliers == expected_grid[0]
        or best.expected_outliers == expected_grid[-1]
    ):
        print(
            "Warning: best (kappa, expected_outliers) lies on grid boundary. "
            "Consider expanding the grid for more confidence."
        )

    print(f"CV tuning finished in {time.perf_counter() - overall_start:.2f}s")
    return best

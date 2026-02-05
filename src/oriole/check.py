from __future__ import annotations

from .error import new_error
from .options.config import Config, endophenotype_mask, endophenotype_names
from .util.dag import is_dag, find_cycle
from .params import Params


def check_config(config: Config) -> None:
    if not config.gwas:
        raise new_error("No GWAS specified.")
    if not config.endophenotypes:
        raise new_error("No endophenotypes specified.")
    gwas_names = {item.name for item in config.gwas}
    trait_names = [item.name for item in config.gwas]
    seen_edges: set[tuple[str, str]] = set()
    edges: list[tuple[str, str]] = []
    for endo in config.endophenotypes:
        if not endo.name:
            raise new_error("Endophenotype name cannot be empty.")
        if "*" in endo.traits:
            continue
        for trait in endo.traits:
            if trait not in gwas_names:
                raise new_error(
                    "Endophenotype {} references unknown trait {}.".format(
                        endo.name, trait
                    )
                )
    for edge in config.trait_edges:
        if edge.parent == edge.child:
            raise new_error("Trait edge cannot have the same parent and child.")
        if edge.parent not in gwas_names or edge.child not in gwas_names:
            raise new_error(
                "Trait edge references unknown trait: {} -> {}.".format(
                    edge.parent, edge.child
                )
            )
        key = (edge.parent, edge.child)
        if key in seen_edges:
            raise new_error(
                "Duplicate trait edge specified: {} -> {}.".format(
                    edge.parent, edge.child
                )
            )
        seen_edges.add(key)
        edges.append(key)
    if edges and not is_dag(trait_names, edges):
        cycle = find_cycle(trait_names, edges)
        if cycle:
            cycle_str = " -> ".join(cycle)
            raise new_error(f"Trait edges must form a DAG (cycle detected: {cycle_str}).")
        raise new_error("Trait edges must form a DAG (cycle detected).")
    if config.outliers.enabled:
        if config.outliers.kappa < 1.0:
            raise new_error("Outlier kappa must be >= 1.0.")
        if config.outliers.expected_outliers_specified:
            n_traits = len(config.gwas)
            if config.outliers.expected_outliers is None:
                raise new_error("Outliers expected_outliers missing.")
            if config.outliers.expected_outliers <= 0.0:
                raise new_error("Outlier expected_outliers must be > 0.")
            if config.outliers.expected_outliers >= n_traits:
                raise new_error(
                    "Outlier expected_outliers must be < number of traits."
                )
        if not (0.0 < config.outliers.pi < 1.0):
            raise new_error("Outlier pi must be in (0, 1).")
        if config.outliers.pi_by_trait:
            if config.outliers.expected_outliers_specified:
                raise new_error(
                    "Outlier expected_outliers cannot be used with pi_by_trait."
                )
            for name, value in config.outliers.pi_by_trait.items():
                if name not in gwas_names:
                    raise new_error(f"Outlier pi_by_trait references unknown trait {name}.")
                if not (0.0 < value < 1.0):
                    raise new_error(
                        f"Outlier pi for trait {name} must be in (0, 1)."
                    )
        if config.outliers.max_enum_traits <= 0:
            raise new_error("Outlier max_enum_traits must be positive.")
    if config.tune_outliers.enabled:
        tune = config.tune_outliers
        if config.outliers.pi_by_trait:
            raise new_error("tune_outliers requires a shared outlier rate (no pi_by_trait).")
        if tune.n_folds < 2:
            raise new_error("tune_outliers.n_folds must be >= 2.")
        if not tune.kappa_grid or not tune.expected_outliers_grid:
            raise new_error(
                "tune_outliers.kappa_grid and expected_outliers_grid must be non-empty."
            )
        for value in tune.kappa_grid:
            if value < 1.0:
                raise new_error("tune_outliers.kappa_grid values must be >= 1.0.")
        n_traits = len(config.gwas)
        for value in tune.expected_outliers_grid:
            if value <= 0.0 or value >= n_traits:
                raise new_error(
                    "tune_outliers.expected_outliers_grid values must be in (0, n_traits)."
                )
        if tune.min_grid_length < 1:
            raise new_error("tune_outliers.min_grid_length must be >= 1.")
        if tune.max_grid_length < tune.min_grid_length:
            raise new_error("tune_outliers.max_grid_length must be >= min_grid_length.")
        if tune.max_expansions < 0:
            raise new_error("tune_outliers.max_expansions must be >= 0.")
        if tune.expansion_factor <= 1.0:
            raise new_error("tune_outliers.expansion_factor must be > 1.0.")
        if tune.boundary_margin < 0.0:
            raise new_error("tune_outliers.boundary_margin must be >= 0.")
        if tune.n_background_sample <= 0:
            raise new_error("tune_outliers.n_background_sample must be positive.")
        if tune.n_negative_sample < 0:
            raise new_error("tune_outliers.n_negative_sample must be >= 0.")
        if not tune.fpr_targets:
            raise new_error("tune_outliers.fpr_targets must be non-empty.")
        for value in tune.fpr_targets:
            if value <= 0.0 or value >= 1.0:
                raise new_error("tune_outliers.fpr_targets must be in (0, 1).")
        if tune.genomewide_ids_file is None and tune.negative_ids_file is None:
            raise new_error(
                "tune_outliers requires genomewide_ids_file (or hard negatives)."
            )


def check_params(config: Config, params: Params) -> None:
    if len(config.gwas) != len(params.trait_names):
        config_names = [item.name for item in config.gwas]
        params_names = params.trait_names
        missing = [name for name in config_names if name not in params_names]
        extra = [name for name in params_names if name not in config_names]
        raise new_error(
            "Number GWAS files ({}) does not match number of traits in params ({}). "
            "Missing in params: {}. Extra in params: {}.".format(
                len(config.gwas),
                len(params.trait_names),
                ", ".join(missing) or "<none>",
                ", ".join(extra) or "<none>",
            )
        )
    for i_trait, (gwas, trait_name) in enumerate(
        zip(config.gwas, params.trait_names)
    ):
        if gwas.name != trait_name:
            raise new_error(
                "Trait name mismatch at index {}: config has '{}' but params has '{}'. "
                "Expected order: [{}].".format(
                    i_trait,
                    gwas.name,
                    trait_name,
                    ", ".join([item.name for item in config.gwas]),
                )
            )
    config_endos = endophenotype_names(config)
    if params.endo_names != config_endos:
        raise new_error(
            "Endophenotype names in params ({}) do not match config ({}).".format(
                params.endo_names, config_endos
            )
        )
    if len(params.betas) != len(params.trait_names):
        raise new_error("Params betas must have one row per trait.")
    for row in params.betas:
        if len(row) != len(params.endo_names):
            raise new_error("Params betas rows must match number of endophenotypes.")
    if len(params.trait_edges) != len(params.trait_names):
        raise new_error("Params trait_edges must have one row per trait.")
    for row in params.trait_edges:
        if len(row) != len(params.trait_names):
            raise new_error("Params trait_edges rows must match number of traits.")
    if len(params.outlier_pis) != len(params.trait_names):
        raise new_error("Params outlier_pis must have one entry per trait.")
    if params.outlier_kappa < 1.0:
        raise new_error("Params outlier_kappa must be >= 1.0.")
    for value in params.outlier_pis:
        if value < 0.0 or value >= 1.0:
            raise new_error("Params outlier_pis must be in [0, 1).")
    mask = endophenotype_mask(config)
    for i_trait, row in enumerate(mask):
        for i_endo, allowed in enumerate(row):
            if not allowed and abs(params.betas[i_trait][i_endo]) > 1e-8:
                raise new_error(
                    "Beta for trait {} and endo {} must be 0 (masked off).".format(
                        params.trait_names[i_trait], params.endo_names[i_endo]
                    )
                )
    trait_index = {name: idx for idx, name in enumerate(params.trait_names)}
    for i_child in range(len(params.trait_edges)):
        for i_parent in range(len(params.trait_edges)):
            if i_child == i_parent:
                if abs(params.trait_edges[i_child][i_parent]) > 1e-8:
                    raise new_error("Trait edge diagonal must be zero.")
                continue
            edge_present = any(
                edge.child == params.trait_names[i_child]
                and edge.parent == params.trait_names[i_parent]
                for edge in config.trait_edges
            )
            if not edge_present and abs(params.trait_edges[i_child][i_parent]) > 1e-8:
                raise new_error(
                    "Trait edge {} -> {} must be 0 (not in config).".format(
                        params.trait_names[i_parent], params.trait_names[i_child]
                    )
                )

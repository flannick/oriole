from __future__ import annotations

from .error import new_error
from .options.config import Config, endophenotype_mask, endophenotype_names
from .util.dag import is_dag
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
        raise new_error("Trait edges must form a DAG (cycle detected).")


def check_params(config: Config, params: Params) -> None:
    if len(config.gwas) != len(params.trait_names):
        raise new_error(
            "Number GWAS files ({}) does not match number of traits in params ({})".format(
                len(config.gwas), len(params.trait_names)
            )
        )
    for i_trait, (gwas, trait_name) in enumerate(
        zip(config.gwas, params.trait_names)
    ):
        if gwas.name != trait_name:
            raise new_error(
                "Trait name in GWAS file {} ({}) does not match trait name in params ({})".format(
                    i_trait, gwas.name, trait_name
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

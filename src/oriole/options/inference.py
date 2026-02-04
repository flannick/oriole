from __future__ import annotations

from ..error import new_error
from .config import Config


def build_outlier_pis(config: Config, trait_names: list[str]) -> list[float]:
    pis = [config.outliers.pi for _ in trait_names]
    if config.outliers.pi_by_trait:
        for name, value in config.outliers.pi_by_trait.items():
            if name in trait_names:
                pis[trait_names.index(name)] = float(value)
    return pis


def resolve_inference(config: Config, inference: str, n_traits: int) -> str:
    method = inference or "auto"
    if method == "auto" and config.outliers.method:
        method = config.outliers.method
    if config.outliers.enabled:
        if method == "auto":
            if n_traits <= config.outliers.max_enum_traits and not config.trait_edges:
                return "analytic"
            return "variational"
        if method == "analytic":
            if config.trait_edges:
                raise new_error(
                    "Analytic enumeration for outliers is only supported without trait_edges."
                )
            if n_traits > config.outliers.max_enum_traits:
                raise new_error(
                    "Analytic enumeration for outliers only supported for <= {} traits; got {}."
                    .format(config.outliers.max_enum_traits, n_traits)
                )
            return "analytic"
        if method in ("variational", "gibbs"):
            return method
        raise new_error(f"Unknown inference method '{method}'.")

    if method in ("auto", "analytic"):
        return "analytic"
    if method == "gibbs":
        return "gibbs"
    if method == "variational":
        print("Note: variational inference is only used with outliers; falling back to analytic.")
        return "analytic"
    raise new_error(f"Unknown inference method '{method}'.")

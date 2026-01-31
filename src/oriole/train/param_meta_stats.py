from __future__ import annotations

from dataclasses import dataclass

from ..error import new_error
from ..math_utils.stats import Stats
from ..math_utils.trident import TridentStats
from ..params import ParamIndex, Params


@dataclass
class Summary:
    n_chains_used: int
    params: Params
    intra_chain_vars: list[float]
    inter_chain_vars: list[float]
    inter_intra_ratios: list[float]
    relative_errors: list[float]
    inter_intra_ratios_mean: float
    relative_errors_mean: float

    def __str__(self) -> str:
        n_traits = self.params.n_traits()
        n_endos = self.params.n_endos()
        lines = [
            f"Chains used: {self.n_chains_used}",
            f"Relative errors mean: {self.relative_errors_mean}",
            f"Inter/intra ratios mean: {self.inter_intra_ratios_mean**0.5}",
            f"{'param':<18} {'value':<18} {'rel.err.':<18} {'inter_chains':<18} "
            f"{'intra_chains':<18} {'ratio':<18}",
        ]
        for i, index in enumerate(ParamIndex.all(n_traits, n_endos)):
            param = self.params[index]
            rel_err = self.relative_errors[i]
            inter_chain_std = self.inter_chain_vars[i] ** 0.5
            intra_chain_std = self.intra_chain_vars[i] ** 0.5
            ratio = self.inter_intra_ratios[i]
            lines.append(
                f"{index.with_trait_name(self.params.trait_names, self.params.endo_names):<18} "
                f"{param:<18} {rel_err:<18} {inter_chain_std:<18} "
                f"{intra_chain_std:<18} {ratio:<18}"
            )
        return "\n".join(lines)


class ParamMetaStats:
    def __init__(
        self,
        n_chains_used: int,
        trait_names: list[str],
        endo_names: list[str],
        params0: list[Params],
        params1: list[Params],
    ):
        n_traits = len(trait_names)
        self.trait_names = trait_names
        self.endo_names = endo_names
        self.stats = []
        for i_chain in range(n_chains_used):
            chain_stats = []
            for index in ParamIndex.all(n_traits, len(endo_names)):
                param0 = params0[i_chain][index]
                param1 = params1[i_chain][index]
                chain_stats.append(TridentStats.new(param0, param1))
            self.stats.append(chain_stats)

    def n_traits(self) -> int:
        return len(self.trait_names)

    def n_chains_used(self) -> int:
        return len(self.stats)

    def add(self, params: list[Params]) -> None:
        n_traits = self.n_traits()
        for i_chain, param in enumerate(params):
            for index in ParamIndex.all(n_traits, len(self.endo_names)):
                i_param = index.get_ordinal(n_traits, len(self.endo_names))
                self.stats[i_chain][i_param].add(param[index])

    def summary(self) -> Summary:
        n_traits = self.n_traits()
        n_chains_used = self.n_chains_used()
        n_endos = len(self.endo_names)
        n_params = ParamIndex.n_params(n_traits, n_endos)
        param_values: list[float] = []
        intra_chain_vars: list[float] = []
        inter_chain_vars: list[float] = []
        inter_intra_ratios: list[float] = []
        relative_errors: list[float] = []
        for index in ParamIndex.all(n_traits, n_endos):
            i_param = index.get_ordinal(n_traits, n_endos)
            inter_mean_stats = Stats()
            inter_var_stats = Stats()
            for i_chain in range(n_chains_used):
                stats = self.stats[i_chain][i_param]
                inter_mean_stats.add(stats.mean())
                inter_var_stats.add(stats.variance())
            param_value = _unwrap_or_not_enough_data(inter_mean_stats.mean())
            intra_chain_var = _unwrap_or_not_enough_data(inter_var_stats.mean())
            inter_chain_var = _unwrap_or_not_enough_data(inter_mean_stats.variance())
            inter_intra_ratio = inter_chain_var / intra_chain_var
            relative_error = (intra_chain_var**0.5 / abs(param_value)) / (n_chains_used**0.5)
            param_values.append(param_value)
            intra_chain_vars.append(intra_chain_var)
            inter_chain_vars.append(inter_chain_var)
            inter_intra_ratios.append(inter_intra_ratio)
            relative_errors.append(relative_error)
        params = Params.from_vec(param_values, self.trait_names, self.endo_names)
        inter_intra_ratios_mean = sum(inter_intra_ratios) / n_params
        relative_errors_mean = sum(relative_errors) / n_params
        return Summary(
            n_chains_used=n_chains_used,
            params=params,
            intra_chain_vars=intra_chain_vars,
            inter_chain_vars=inter_chain_vars,
            inter_intra_ratios=inter_intra_ratios,
            relative_errors=relative_errors,
            inter_intra_ratios_mean=inter_intra_ratios_mean,
            relative_errors_mean=relative_errors_mean,
        )


def _unwrap_or_not_enough_data(value: float | None) -> float:
    if value is None:
        raise new_error("Not enough data")
    return value

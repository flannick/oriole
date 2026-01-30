from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import re

from ..error import new_error, for_context, for_file
from ..math.matrix import matrix_fill
from ..options.action import Action
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..options.config import Config, GwasConfig
from .gwas import GwasReader, GwasRecord, GwasCols, default_cols

DELIM_LIST = [";", "\t", ",", " "]


@dataclass
class Meta:
    trait_names: list[str]
    var_ids: list[str]
    endo_names: list[str]

    def n_data_points(self) -> int:
        return len(self.var_ids)

    def n_traits(self) -> int:
        return len(self.trait_names)

    def n_endos(self) -> int:
        return len(self.endo_names)


@dataclass
class GwasData:
    meta: Meta
    betas: np.ndarray
    ses: np.ndarray

    def n_data_points(self) -> int:
        return self.meta.n_data_points()

    def n_traits(self) -> int:
        return self.meta.n_traits()

    def only_data_point(self, i_row: int) -> tuple["GwasData", list[int]]:
        var_id = self.meta.var_ids[i_row]
        is_col = [
            i_col
            for i_col in range(self.n_traits())
            if np.isfinite(self.betas[i_row, i_col]) and np.isfinite(self.ses[i_row, i_col])
        ]
        trait_names = [self.meta.trait_names[i_col] for i_col in is_col]
        betas = self.betas[i_row, is_col].reshape(1, -1)
        ses = self.ses[i_row, is_col].reshape(1, -1)
        meta = Meta(trait_names=trait_names, var_ids=[var_id], endo_names=self.meta.endo_names)
        return GwasData(meta=meta, betas=betas, ses=ses), is_col

    def __str__(self) -> str:
        lines = [default_cols.VAR_ID]
        for trait_name in self.meta.trait_names:
            lines.append(f"beta_{trait_name}")
            lines.append(f"se_{trait_name}")
        out = ["\t".join(lines)]
        for i_data_point, var_id in enumerate(self.meta.var_ids):
            row = [var_id]
            for i_trait in range(self.n_traits()):
                row.append(str(self.betas[i_data_point, i_trait]))
                row.append(str(self.ses[i_data_point, i_trait]))
            out.append("\t".join(row))
        return "\n".join(out)


@dataclass
class Weights:
    weights: list[float]
    sum: float

    @classmethod
    def new(cls, n_data_points: int) -> "Weights":
        return cls(weights=[], sum=0.0)

    def add(self, weight: float) -> None:
        self.weights.append(weight)
        self.sum += weight


@dataclass
class LoadedData:
    gwas_data: GwasData
    weights: Weights


@dataclass
class BetaSe:
    beta: float
    se: float


@dataclass
class IdData:
    beta_se_list: list[BetaSe]
    weight: float


def load_data(config: Config, action: Action) -> LoadedData:
    n_traits = len(config.gwas)
    if action == Action.TRAIN:
        beta_se_by_id = load_ids(config.train.ids_file, n_traits)
    else:
        beta_se_by_id = {}

    trait_names: list[str] = []
    for i_trait, gwas in enumerate(config.gwas):
        trait_names.append(gwas.name)
        load_gwas(beta_se_by_id, gwas, n_traits, i_trait, action)

    n_data_points = len(beta_se_by_id)
    var_ids: list[str] = []
    betas = matrix_fill(n_data_points, n_traits, lambda _i, _j: np.nan)
    ses = matrix_fill(n_data_points, n_traits, lambda _i, _j: np.nan)
    weights = Weights.new(n_data_points)

    for i_data_point, (var_id, id_data) in enumerate(sorted(beta_se_by_id.items())):
        var_ids.append(var_id)
        weights.add(id_data.weight)
        for i_trait, beta_se in enumerate(id_data.beta_se_list):
            betas[i_data_point, i_trait] = beta_se.beta
            ses[i_data_point, i_trait] = beta_se.se

    if action == Action.TRAIN:
        for i_data_point, var_id in enumerate(var_ids):
            for i_trait, trait_name in enumerate(trait_names):
                if np.isnan(betas[i_data_point, i_trait]):
                    raise new_error(
                        f"Missing beta for trait {trait_name} for var id {var_id}"
                    )
                if np.isnan(ses[i_data_point, i_trait]):
                    raise new_error(
                        f"Missing se for trait {trait_name} for var id {var_id}"
                    )

    endo_names = [item.name for item in config.endophenotypes]
    meta = Meta(trait_names=trait_names, var_ids=var_ids, endo_names=endo_names)
    gwas_data = GwasData(meta=meta, betas=betas, ses=ses)
    return LoadedData(gwas_data=gwas_data, weights=weights)


def load_ids(ids_file: str, n_traits: int) -> Dict[str, IdData]:
    beta_se_by_id: Dict[str, IdData] = {}
    this_might_still_be_header = True
    splitter = re.compile(r"[;\t, ]+")
    try:
        with open(ids_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = [part for part in splitter.split(line) if part != ""]
                if len(parts) == 1:
                    id_value = parts[0]
                    weight_str = None
                else:
                    id_value, weight_str = parts[0], parts[1]
                if id_value:
                    beta_se_list = new_beta_se_list(n_traits)
                    if weight_str is None:
                        weight = 1.0
                        this_might_still_be_header = False
                        beta_se_by_id[id_value] = IdData(beta_se_list, weight)
                    else:
                        try:
                            weight = float(weight_str)
                        except ValueError as exc:
                            if this_might_still_be_header:
                                continue
                            raise for_context(
                                f"Error parsing weight for id {id_value}", exc
                            ) from exc
                        this_might_still_be_header = False
                        if weight < 0.0:
                            raise new_error(f"Negative weight ({weight}) for id {id_value}")
                        beta_se_by_id[id_value] = IdData(beta_se_list, weight)
    except Exception as exc:
        raise for_file(ids_file, exc) from exc
    return beta_se_by_id


def load_gwas(
    beta_se_by_id: Dict[str, IdData],
    gwas_config: "GwasConfig",
    n_traits: int,
    i_trait: int,
    action: Action,
) -> None:
    file = gwas_config.file
    cols = gwas_config.cols or GwasCols(
        id=default_cols.VAR_ID, effect=default_cols.BETA, se=default_cols.SE
    )
    try:
        with open(file, "r", encoding="utf-8") as handle:
            reader = GwasReader(handle, cols)
            for record in reader:
                var_id = record.var_id
                beta = record.beta
                se = record.se
                if var_id in beta_se_by_id:
                    beta_se_by_id[var_id].beta_se_list[i_trait] = BetaSe(beta=beta, se=se)
                elif action == Action.CLASSIFY:
                    beta_se_list = new_beta_se_list(n_traits)
                    beta_se_list[i_trait] = BetaSe(beta=beta, se=se)
                    beta_se_by_id[var_id] = IdData(beta_se_list, 1.0)
    except Exception as exc:
        raise for_context(file, exc) from exc


def new_beta_se_list(n_traits: int) -> list[BetaSe]:
    return [BetaSe(beta=float("nan"), se=float("nan")) for _ in range(n_traits)]

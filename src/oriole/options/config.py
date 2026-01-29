from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib
import tomli_w

from ..error import MocasaError, ErrorKind
from ..params import ParamsOverride
from ..data.gwas import GwasCols


@dataclass
class GwasConfig:
    name: str
    file: str
    cols: Optional[GwasCols] = None


@dataclass
class FilesConfig:
    trace: Optional[str] = None
    params: str = ""


@dataclass
class TrainConfig:
    ids_file: str
    n_steps_burn_in: int
    n_samples_per_iteration: int
    n_iterations_per_round: int
    n_rounds: int
    normalize_mu_to_one: bool
    params_trace_file: Optional[str] = None
    t_pinned: Optional[bool] = None


@dataclass
class ClassifyConfig:
    params_override: Optional[ParamsOverride]
    n_steps_burn_in: int
    n_samples: int
    out_file: str
    trace_ids: Optional[list[str]] = None
    t_pinned: Optional[bool] = None


@dataclass
class Config:
    files: FilesConfig
    gwas: list[GwasConfig]
    train: TrainConfig
    classify: ClassifyConfig


def _gwas_cols_from_dict(value: dict | None) -> Optional[GwasCols]:
    if value is None:
        return None
    return GwasCols(id=value["id"], effect=value["effect"], se=value["se"])


def _params_override_from_dict(value: dict | None) -> Optional[ParamsOverride]:
    if value is None:
        return None
    return ParamsOverride(mu=value.get("mu"), tau=value.get("tau"))


def load_config(path: str) -> Config:
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except Exception as exc:
        raise MocasaError(ErrorKind.TOML_DE, str(exc)) from exc

    files = FilesConfig(**data["files"])
    gwas = [
        GwasConfig(
            name=item["name"],
            file=item["file"],
            cols=_gwas_cols_from_dict(item.get("cols")),
        )
        for item in data["gwas"]
    ]
    train = TrainConfig(**data["train"])
    classify_data = data["classify"].copy()
    classify_data["params_override"] = _params_override_from_dict(
        classify_data.get("params_override")
    )
    classify = ClassifyConfig(**classify_data)
    return Config(files=files, gwas=gwas, train=train, classify=classify)


def dump_config(config: Config) -> str:
    def gwas_to_dict(item: GwasConfig) -> dict:
        cols = None
        if item.cols is not None:
            cols = {"id": item.cols.id, "effect": item.cols.effect, "se": item.cols.se}
        return {"name": item.name, "file": item.file, "cols": cols}

    def classify_to_dict(item: ClassifyConfig) -> dict:
        data = {
            "params_override": None,
            "n_steps_burn_in": item.n_steps_burn_in,
            "n_samples": item.n_samples,
            "out_file": item.out_file,
            "trace_ids": item.trace_ids,
            "t_pinned": item.t_pinned,
        }
        if item.params_override is not None:
            data["params_override"] = {
                "mu": item.params_override.mu,
                "tau": item.params_override.tau,
            }
        return data

    data = {
        "files": {"trace": config.files.trace, "params": config.files.params},
        "gwas": [gwas_to_dict(item) for item in config.gwas],
        "train": {
            "ids_file": config.train.ids_file,
            "n_steps_burn_in": config.train.n_steps_burn_in,
            "n_samples_per_iteration": config.train.n_samples_per_iteration,
            "n_iterations_per_round": config.train.n_iterations_per_round,
            "n_rounds": config.train.n_rounds,
            "normalize_mu_to_one": config.train.normalize_mu_to_one,
            "params_trace_file": config.train.params_trace_file,
            "t_pinned": config.train.t_pinned,
        },
        "classify": classify_to_dict(config.classify),
    }
    try:
        return tomli_w.dumps(data)
    except Exception as exc:
        raise MocasaError(ErrorKind.TOML_SER, str(exc)) from exc

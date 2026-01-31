from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
class EndophenotypeConfig:
    name: str
    traits: list[str]


@dataclass
class TraitEdgeConfig:
    parent: str
    child: str


@dataclass
class Config:
    files: FilesConfig
    gwas: list[GwasConfig]
    endophenotypes: list[EndophenotypeConfig]
    trait_edges: list[TraitEdgeConfig]
    train: TrainConfig
    classify: ClassifyConfig


def _gwas_cols_from_dict(value: dict | None) -> Optional[GwasCols]:
    if value is None:
        return None
    return GwasCols(id=value["id"], effect=value["effect"], se=value["se"])


def _params_override_from_dict(value: dict | None) -> Optional[ParamsOverride]:
    if value is None:
        return None
    return ParamsOverride(
        mu=value.get("mu"),
        tau=value.get("tau"),
        mus=value.get("mus"),
        taus=value.get("taus"),
        mus_by_name=value.get("mus_by_name"),
        taus_by_name=value.get("taus_by_name"),
    )


def _endophenotypes_from_dicts(value: list[dict] | None) -> list[EndophenotypeConfig]:
    if not value:
        return [EndophenotypeConfig(name="E", traits=["*"])]
    return [
        EndophenotypeConfig(name=item["name"], traits=list(item["traits"]))
        for item in value
    ]


def load_config(path: str) -> Config:
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except Exception as exc:
        raise MocasaError(ErrorKind.TOML_DE, str(exc)) from exc

    files_data = data.get("files", {})
    params_path = files_data.get("params")
    if not params_path:
        params_path = str((Path(path).parent / "params_out.json").as_posix())
    files = FilesConfig(trace=files_data.get("trace"), params=params_path)
    gwas = [
        GwasConfig(
            name=item["name"],
            file=item["file"],
            cols=_gwas_cols_from_dict(item.get("cols")),
        )
        for item in data["gwas"]
    ]
    endophenotypes = _endophenotypes_from_dicts(data.get("endophenotypes"))
    trait_edges = [
        TraitEdgeConfig(parent=item["parent"], child=item["child"])
        for item in data.get("trait_edges", [])
    ]
    train_data = data.get("train")
    if not train_data or "ids_file" not in train_data:
        raise MocasaError(ErrorKind.TOML_DE, "Missing [train].ids_file in config.")
    train = TrainConfig(
        ids_file=train_data["ids_file"],
        n_steps_burn_in=train_data.get("n_steps_burn_in", 1000),
        n_samples_per_iteration=train_data.get("n_samples_per_iteration", 100),
        n_iterations_per_round=train_data.get("n_iterations_per_round", 10),
        n_rounds=train_data.get("n_rounds", 1),
        normalize_mu_to_one=train_data.get("normalize_mu_to_one", True),
        params_trace_file=train_data.get("params_trace_file"),
        t_pinned=train_data.get("t_pinned"),
    )
    classify_data = data.get("classify")
    if classify_data is None:
        classify_data = {}
    else:
        classify_data = classify_data.copy()
    classify_data["params_override"] = _params_override_from_dict(
        classify_data.get("params_override")
    )
    classify_data.setdefault("n_steps_burn_in", 1000)
    classify_data.setdefault("n_samples", 1000)
    classify_data.setdefault("out_file", "classify_out.tsv")
    classify_data.setdefault("trace_ids", [])
    classify_data.setdefault("t_pinned", None)
    classify = ClassifyConfig(**classify_data)
    return Config(
        files=files,
        gwas=gwas,
        endophenotypes=endophenotypes,
        trait_edges=trait_edges,
        train=train,
        classify=classify,
    )


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
        "endophenotypes": [
            {"name": item.name, "traits": item.traits} for item in config.endophenotypes
        ],
        "trait_edges": [
            {"parent": item.parent, "child": item.child} for item in config.trait_edges
        ],
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


def endophenotype_names(config: Config) -> list[str]:
    return [item.name for item in config.endophenotypes]


def endophenotype_mask(config: Config) -> list[list[bool]]:
    trait_names = [item.name for item in config.gwas]
    mask: list[list[bool]] = []
    for trait in trait_names:
        trait_mask: list[bool] = []
        for endo in config.endophenotypes:
            if "*" in endo.traits:
                trait_mask.append(True)
            else:
                trait_mask.append(trait in endo.traits)
        mask.append(trait_mask)
    return mask

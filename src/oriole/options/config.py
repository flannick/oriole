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
class DataAccessConfig:
    gwas_base_uri: Optional[str] = None


@dataclass
class TrainConfig:
    ids_file: str
    n_steps_burn_in: int
    n_samples_per_iteration: int
    n_iterations_per_round: int
    n_rounds: int
    normalize_mu_to_one: bool
    learn_mu_tau: bool
    mu: float
    tau: float
    mu_specified: bool
    tau_specified: bool
    params_trace_file: Optional[str] = None
    t_pinned: Optional[bool] = None
    plot_convergence_out_file: Optional[str] = None
    plot_cv_out_file: Optional[str] = None
    early_stop: bool = False
    early_stop_patience: int = 3
    early_stop_rel_tol: float = 1e-4
    early_stop_obj_tol: float = 1e-5
    early_stop_min_iters: int = 5


@dataclass
class ClassifyConfig:
    params_override: Optional[ParamsOverride]
    n_steps_burn_in: int
    n_samples: int
    out_file: str
    trace_ids: Optional[list[str]] = None
    t_pinned: Optional[bool] = None
    write_full: bool = True
    gwas_ssf_out_file: Optional[str] = None
    gwas_ssf_guess_fields: bool = True
    gwas_ssf_variant_id_order: str = "effect_other"
    mu_specified: bool = False
    tau_specified: bool = False
    mu: float = 0.0
    tau: float = 1e6


@dataclass
class EndophenotypeConfig:
    name: str
    traits: list[str]


@dataclass
class OutliersConfig:
    enabled: bool
    kappa: float
    expected_outliers: Optional[float]
    pi: float
    pi_by_trait: dict[str, float] | None
    max_enum_traits: int
    method: str | None
    kappa_specified: bool
    pi_specified: bool
    expected_outliers_specified: bool


@dataclass
class TuneOutliersConfig:
    enabled: bool
    mode: str
    n_folds: int
    seed: int
    kappa_grid: list[float]
    expected_outliers_grid: list[float]
    kappa_grid_specified: bool
    expected_outliers_grid_specified: bool
    min_grid_length: int
    max_grid_length: int
    genomewide_ids_file: Optional[str]
    negative_ids_file: Optional[str]
    n_background_sample: int
    n_negative_sample: int
    fpr_targets: list[float]
    lambda_hard: float
    expand_on_boundary: bool
    max_expansions: int
    expansion_factor: float
    force_expansions: bool
    boundary_margin: float


@dataclass
class TraitEdgeConfig:
    parent: str
    child: str


@dataclass
class Config:
    files: FilesConfig
    data_access: DataAccessConfig
    gwas: list[GwasConfig]
    endophenotypes: list[EndophenotypeConfig]
    trait_edges: list[TraitEdgeConfig]
    outliers: OutliersConfig
    tune_outliers: TuneOutliersConfig
    train: TrainConfig
    classify: ClassifyConfig


def _gwas_cols_from_dict(value: dict | None) -> Optional[GwasCols]:
    if value is None:
        return None
    return GwasCols(
        id=value["id"],
        effect=value["effect"],
        se=value["se"],
        chrom=value.get("chrom"),
        pos=value.get("pos"),
        effect_allele=value.get("effect_allele"),
        other_allele=value.get("other_allele"),
        eaf=value.get("eaf"),
        rsid=value.get("rsid"),
    )


def _params_override_from_dict(value: dict | None) -> Optional[ParamsOverride]:
    if value is None:
        return None
    override = ParamsOverride(
        mu=value.get("mu"),
        tau=value.get("tau"),
        mus=value.get("mus"),
        taus=value.get("taus"),
        mus_by_name=value.get("mus_by_name"),
        taus_by_name=value.get("taus_by_name"),
    )
    if (
        override.mu is None
        and override.tau is None
        and override.mus is None
        and override.taus is None
        and not override.mus_by_name
        and not override.taus_by_name
    ):
        return None
    return override


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
    data_access_data = data.get("data_access", {})
    data_access = DataAccessConfig(
        gwas_base_uri=data_access_data.get("gwas_base_uri")
    )
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
    outliers_data = data.get("outliers", {})
    kappa_specified = "kappa" in outliers_data
    pi_specified = "pi" in outliers_data or "pi_by_trait" in outliers_data
    expected_outliers_specified = "expected_outliers" in outliers_data
    if expected_outliers_specified and pi_specified:
        raise MocasaError(
            ErrorKind.TOML_DE,
            "outliers.expected_outliers is mutually exclusive with outliers.pi/pi_by_trait.",
        )
    n_traits = len(gwas)
    expected_outliers_value = None
    if expected_outliers_specified:
        expected_outliers_value = float(outliers_data["expected_outliers"])
        if expected_outliers_value <= 0.0:
            raise MocasaError(
                ErrorKind.TOML_DE,
                "outliers.expected_outliers must be > 0.",
            )
        pi_value = expected_outliers_value / max(1, n_traits)
    else:
        pi_value = float(outliers_data.get("pi", DEFAULT_OUTLIER_PI))
    outliers = OutliersConfig(
        enabled=bool(outliers_data.get("enabled", False)),
        kappa=float(outliers_data.get("kappa", DEFAULT_OUTLIER_KAPPA)),
        expected_outliers=expected_outliers_value,
        pi=pi_value,
        pi_by_trait=outliers_data.get("pi_by_trait"),
        max_enum_traits=int(outliers_data.get("max_enum_traits", 12)),
        method=outliers_data.get("method"),
        kappa_specified=kappa_specified,
        pi_specified=pi_specified,
        expected_outliers_specified=expected_outliers_specified,
    )
    tune_data = data.get("tune_outliers", {})
    default_kappa_grid = [DEFAULT_OUTLIER_KAPPA]
    default_expected_outliers_grid = [DEFAULT_OUTLIER_PI * max(1, n_traits)]
    kappa_grid_specified = "kappa_grid" in tune_data
    expected_outliers_grid_specified = (
        "expected_outliers_grid" in tune_data or "pi_grid" in tune_data
    )
    kappa_grid = [float(v) for v in tune_data.get("kappa_grid", [])]
    expected_outliers_grid = [
        float(v) for v in tune_data.get("expected_outliers_grid", [])
    ]
    if not expected_outliers_grid and "pi_grid" in tune_data:
        expected_outliers_grid = [
            float(v) * max(1, n_traits) for v in tune_data.get("pi_grid", [])
        ]
    if not kappa_grid:
        kappa_grid = list(default_kappa_grid)
    if not expected_outliers_grid:
        expected_outliers_grid = list(default_expected_outliers_grid)
    tune_mode = str(tune_data.get("mode", "off"))
    if tune_mode == "on":
        tune_mode = "auto"
    tune_outliers = TuneOutliersConfig(
        enabled=bool(tune_data.get("enabled", False)),
        mode=tune_mode,
        n_folds=int(tune_data.get("n_folds", 5)),
        seed=int(tune_data.get("seed", 1)),
        kappa_grid=kappa_grid,
        expected_outliers_grid=expected_outliers_grid,
        kappa_grid_specified=kappa_grid_specified,
        expected_outliers_grid_specified=expected_outliers_grid_specified,
        min_grid_length=int(tune_data.get("min_grid_length", 10)),
        max_grid_length=int(tune_data.get("max_grid_length", 30)),
        genomewide_ids_file=tune_data.get("genomewide_ids_file"),
        negative_ids_file=tune_data.get("negative_ids_file"),
        n_background_sample=int(tune_data.get("n_background_sample", 100000)),
        n_negative_sample=int(tune_data.get("n_negative_sample", 50000)),
        fpr_targets=[float(v) for v in tune_data.get("fpr_targets", [1e-3])],
        lambda_hard=float(tune_data.get("lambda_hard", 0.0)),
        expand_on_boundary=bool(tune_data.get("expand_on_boundary", True)),
        max_expansions=int(tune_data.get("max_expansions", 0)),
        expansion_factor=float(tune_data.get("expansion_factor", 2.0)),
        force_expansions=bool(tune_data.get("force_expansions", False)),
        boundary_margin=float(tune_data.get("boundary_margin", 0.01)),
    )

    train_data = data.get("train")
    if not train_data or "ids_file" not in train_data:
        raise MocasaError(ErrorKind.TOML_DE, "Missing [train].ids_file in config.")
    mu_specified = "mu" in train_data
    tau_specified = "tau" in train_data
    train = TrainConfig(
        ids_file=train_data["ids_file"],
        n_steps_burn_in=train_data.get("n_steps_burn_in", 1000),
        n_samples_per_iteration=train_data.get("n_samples_per_iteration", 100),
        n_iterations_per_round=train_data.get("n_iterations_per_round", 10),
        n_rounds=train_data.get("n_rounds", 1),
        normalize_mu_to_one=train_data.get("normalize_mu_to_one", False),
        learn_mu_tau=bool(train_data.get("learn_mu_tau", False)),
        mu=float(train_data.get("mu", 0.0)),
        tau=float(train_data.get("tau", 1.0)),
        mu_specified=mu_specified,
        tau_specified=tau_specified,
        params_trace_file=train_data.get("params_trace_file"),
        t_pinned=train_data.get("t_pinned"),
        plot_convergence_out_file=train_data.get("plot_convergence_out_file"),
        plot_cv_out_file=train_data.get("plot_cv_out_file"),
        early_stop=bool(train_data.get("early_stop", False)),
        early_stop_patience=int(train_data.get("early_stop_patience", 3)),
        early_stop_rel_tol=float(train_data.get("early_stop_rel_tol", 1e-4)),
        early_stop_obj_tol=float(train_data.get("early_stop_obj_tol", 1e-5)),
        early_stop_min_iters=int(train_data.get("early_stop_min_iters", 5)),
    )
    classify_data = data.get("classify")
    if classify_data is None:
        classify_data = {}
    else:
        classify_data = classify_data.copy()
    classify_mu_specified = "mu" in classify_data
    classify_tau_specified = "tau" in classify_data
    classify_mu = float(classify_data.get("mu", 0.0))
    classify_tau = float(classify_data.get("tau", 1e6))
    classify_data["params_override"] = _params_override_from_dict(
        classify_data.get("params_override")
    )
    if classify_data["params_override"] is None and (classify_mu_specified or classify_tau_specified):
        classify_data["params_override"] = ParamsOverride(
            mu=classify_mu if classify_mu_specified else None,
            tau=classify_tau if classify_tau_specified else None,
        )
    classify_data.setdefault("n_steps_burn_in", 1000)
    classify_data.setdefault("n_samples", 1000)
    classify_data.setdefault("out_file", "classify_out.tsv")
    classify_data.setdefault("write_full", True)
    classify_data.setdefault("gwas_ssf_out_file", None)
    classify_data.setdefault("gwas_ssf_guess_fields", True)
    classify_data.setdefault("gwas_ssf_variant_id_order", "effect_other")
    classify_data.setdefault("trace_ids", [])
    classify_data.setdefault("t_pinned", None)
    classify_data.setdefault("mu_specified", classify_mu_specified)
    classify_data.setdefault("tau_specified", classify_tau_specified)
    classify_data.setdefault("mu", classify_mu)
    classify_data.setdefault("tau", classify_tau)
    classify = ClassifyConfig(**classify_data)
    return Config(
        files=files,
        data_access=data_access,
        gwas=gwas,
        endophenotypes=endophenotypes,
        trait_edges=trait_edges,
        outliers=outliers,
        tune_outliers=tune_outliers,
        train=train,
        classify=classify,
    )


def dump_config(config: Config) -> str:
    def gwas_to_dict(item: GwasConfig) -> dict:
        cols = None
        if item.cols is not None:
            cols = {
                "id": item.cols.id,
                "effect": item.cols.effect,
                "se": item.cols.se,
                "chrom": item.cols.chrom,
                "pos": item.cols.pos,
                "effect_allele": item.cols.effect_allele,
                "other_allele": item.cols.other_allele,
                "eaf": item.cols.eaf,
                "rsid": item.cols.rsid,
            }
        return {"name": item.name, "file": item.file, "cols": cols}

    def classify_to_dict(item: ClassifyConfig) -> dict:
        data = {
            "params_override": None,
            "n_steps_burn_in": item.n_steps_burn_in,
            "n_samples": item.n_samples,
            "out_file": item.out_file,
            "write_full": item.write_full,
            "gwas_ssf_out_file": item.gwas_ssf_out_file,
            "gwas_ssf_guess_fields": item.gwas_ssf_guess_fields,
            "gwas_ssf_variant_id_order": item.gwas_ssf_variant_id_order,
            "trace_ids": item.trace_ids,
            "t_pinned": item.t_pinned,
            "mu": item.mu,
            "tau": item.tau,
        }
        if item.params_override is not None:
            data["params_override"] = {
                "mu": item.params_override.mu,
                "tau": item.params_override.tau,
            }
        return data

    data = {
        "files": {"trace": config.files.trace, "params": config.files.params},
        "data_access": {"gwas_base_uri": config.data_access.gwas_base_uri},
        "gwas": [gwas_to_dict(item) for item in config.gwas],
        "endophenotypes": [
            {"name": item.name, "traits": item.traits} for item in config.endophenotypes
        ],
        "trait_edges": [
            {"parent": item.parent, "child": item.child} for item in config.trait_edges
        ],
        "outliers": {
            "enabled": config.outliers.enabled,
            "kappa": config.outliers.kappa,
            "pi": config.outliers.pi,
            "pi_by_trait": config.outliers.pi_by_trait,
            "max_enum_traits": config.outliers.max_enum_traits,
            "method": config.outliers.method,
        },
        "tune_outliers": {
            "enabled": config.tune_outliers.enabled,
            "mode": config.tune_outliers.mode,
            "n_folds": config.tune_outliers.n_folds,
            "seed": config.tune_outliers.seed,
            "kappa_grid": config.tune_outliers.kappa_grid,
            "expected_outliers_grid": config.tune_outliers.expected_outliers_grid,
            "kappa_grid_specified": config.tune_outliers.kappa_grid_specified,
            "expected_outliers_grid_specified": config.tune_outliers.expected_outliers_grid_specified,
            "min_grid_length": config.tune_outliers.min_grid_length,
            "max_grid_length": config.tune_outliers.max_grid_length,
            "genomewide_ids_file": config.tune_outliers.genomewide_ids_file,
            "negative_ids_file": config.tune_outliers.negative_ids_file,
            "n_background_sample": config.tune_outliers.n_background_sample,
            "n_negative_sample": config.tune_outliers.n_negative_sample,
            "fpr_targets": config.tune_outliers.fpr_targets,
            "lambda_hard": config.tune_outliers.lambda_hard,
            "expand_on_boundary": config.tune_outliers.expand_on_boundary,
            "max_expansions": config.tune_outliers.max_expansions,
            "expansion_factor": config.tune_outliers.expansion_factor,
            "force_expansions": config.tune_outliers.force_expansions,
            "boundary_margin": config.tune_outliers.boundary_margin,
        },
        "train": {
            "ids_file": config.train.ids_file,
            "n_steps_burn_in": config.train.n_steps_burn_in,
            "n_samples_per_iteration": config.train.n_samples_per_iteration,
            "n_iterations_per_round": config.train.n_iterations_per_round,
            "n_rounds": config.train.n_rounds,
            "normalize_mu_to_one": config.train.normalize_mu_to_one,
            "learn_mu_tau": config.train.learn_mu_tau,
            "mu": config.train.mu,
            "tau": config.train.tau,
            "params_trace_file": config.train.params_trace_file,
            "t_pinned": config.train.t_pinned,
            "plot_convergence_out_file": config.train.plot_convergence_out_file,
            "plot_cv_out_file": config.train.plot_cv_out_file,
            "early_stop": config.train.early_stop,
            "early_stop_patience": config.train.early_stop_patience,
            "early_stop_rel_tol": config.train.early_stop_rel_tol,
            "early_stop_obj_tol": config.train.early_stop_obj_tol,
            "early_stop_min_iters": config.train.early_stop_min_iters,
        },
        "classify": classify_to_dict(config.classify),
    }
    if config.outliers.expected_outliers_specified:
        data["outliers"]["expected_outliers"] = config.outliers.expected_outliers
        data["outliers"].pop("pi", None)
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
DEFAULT_OUTLIER_KAPPA = 4.0
DEFAULT_OUTLIER_PI = 0.16

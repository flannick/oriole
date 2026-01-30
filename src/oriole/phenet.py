from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .data.gwas import GwasCols
from .error import new_error, for_file
from .options.cli import ImportPhenetOptions
from .options.config import (
    ClassifyConfig,
    Config,
    EndophenotypeConfig,
    FilesConfig,
    GwasConfig,
    TrainConfig,
    dump_config,
)
from .params import Params, ParamsOverride, write_params_to_file


class defaults:
    class train:
        N_STEPS_BURN_IN = 10000
        N_SAMPLES_PER_ITERATION = 100
        N_ITERATIONS_PER_ROUND = 1000
        N_ROUNDS = 10000

    class classify:
        N_STEPS_BURN_IN = 10000
        N_SAMPLES = 100000


class keys:
    VAR_ID_FILE = "var_id_file"
    CONFIG_FILE = "config_file"
    OUTPUT_FILE = "output_file"
    DECLARE = "declare"
    TRAIT = "trait"
    ENDO = "endo"
    FILE = "file"
    ID_COL = "id_col"
    EFFECT_COL = "effect_col"
    SE_COL = "se_col"
    BETA = "beta"
    VAR = "var"
    MEAN = "mean"


@dataclass
class PhenetOpts:
    var_id_file: str
    config_files: list[str]
    output_file: str | None


class ConfigBuilder:
    def __init__(self) -> None:
        self.trait_names: list[str] = []
        self.endo_name: str | None = None
        self.files: Dict[str, str] = {}
        self.id_cols: Dict[str, str] = {}
        self.effect_cols: Dict[str, str] = {}
        self.se_cols: Dict[str, str] = {}
        self.betas: Dict[str, float] = {}
        self.vars: Dict[str, float] = {}
        self.means: Dict[str, float] = {}

    def read_phenet_config(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            self.parse_phenet_config(handle)

    def read_phenet_config_optional(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                self.parse_phenet_config(handle)
        except Exception as exc:
            print(exc)
            print("Since this file was optional, we proceed.")

    def parse_phenet_config(self, handle) -> None:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise new_error(f"Cannot parse line: '{line}'.")
            part1, part2, part3 = parts
            if part2 == keys.DECLARE and part3 == keys.TRAIT:
                if part1 not in self.trait_names:
                    self.trait_names.append(part1)
            elif part2 == keys.DECLARE and part3 == keys.ENDO:
                self.endo_name = part1
            elif part2 == keys.FILE:
                self.files[part1] = part3
            elif part2 == keys.ID_COL:
                self.id_cols[part1] = part3
            elif part2 == keys.EFFECT_COL:
                self.effect_cols[part1] = part3
            elif part2 == keys.SE_COL:
                self.se_cols[part1] = part3
            elif part2 == keys.BETA:
                self.betas[part1] = float(part3)
            elif part2 == keys.VAR:
                self.vars[part1] = float(part3)
            elif part2 == keys.MEAN:
                self.means[part1] = float(part3)
            else:
                print(f"Ignoring line '{line}'.")

    def report(self) -> None:
        print(f"Trait names: {', '.join(self.trait_names)}")
        if self.endo_name:
            print(f"Endo name: {self.endo_name}")
        for trait_name in self.trait_names:
            file = self.files.get(trait_name, "<no file>")
            id_col = self.id_cols.get(trait_name, "<no id col>")
            effect_col = self.effect_cols.get(trait_name, "<no effect col>")
            se_col = self.se_cols.get(trait_name, "<no se col>")
            print(f"{trait_name}\t{id_col}\t{effect_col}\t{se_col}\t{file}")

    def build_gwas_configs(self) -> list[GwasConfig]:
        gwas_configs: list[GwasConfig] = []
        default_cols = GwasCols(id="VAR_ID", effect="BETA", se="SE")
        for name in self.trait_names:
            if name not in self.files:
                raise new_error(f"No file specified for {name}")
            id_col = self.id_cols.get(name, default_cols.id)
            effect_col = self.effect_cols.get(name, default_cols.effect)
            se_col = self.se_cols.get(name, default_cols.se)
            cols = GwasCols(id=id_col, effect=effect_col, se=se_col)
            gwas_configs.append(GwasConfig(name=name, file=self.files[name], cols=cols))
        return gwas_configs

    def build_config(self, options: ImportPhenetOptions, phenet_opts: PhenetOpts) -> Config:
        files = FilesConfig(trace=None, params=options.params_file)
        gwas = self.build_gwas_configs()
        ids_file = phenet_opts.var_id_file
        train = TrainConfig(
            ids_file=ids_file,
            n_steps_burn_in=defaults.train.N_STEPS_BURN_IN,
            n_samples_per_iteration=defaults.train.N_SAMPLES_PER_ITERATION,
            n_iterations_per_round=defaults.train.N_ITERATIONS_PER_ROUND,
            n_rounds=defaults.train.N_ROUNDS,
            normalize_mu_to_one=True,
            params_trace_file=None,
            t_pinned=None,
        )
        classify = ClassifyConfig(
            params_override=None,
            n_steps_burn_in=defaults.classify.N_STEPS_BURN_IN,
            n_samples=defaults.classify.N_SAMPLES,
            out_file=options.out_file,
            trace_ids=None,
            t_pinned=None,
        )
        endo_name = self.endo_name or "E"
        endophenotypes = [EndophenotypeConfig(name=endo_name, traits=["*"])]
        return Config(
            files=files,
            gwas=gwas,
            endophenotypes=endophenotypes,
            trait_edges=[],
            train=train,
            classify=classify,
        )

    def got_some_params(self) -> bool:
        return bool(self.betas or self.vars or self.means)

    def build_params(self) -> Params:
        if not self.endo_name:
            raise new_error("No name for endo phenotype specified.")
        trait_names = self.trait_names
        mu = num_or_error(self.means, self.endo_name, keys.MEAN)
        tau = num_or_error(self.vars, self.endo_name, keys.VAR) ** 0.5
        betas: list[float] = []
        sigmas: list[float] = []
        for trait_name in trait_names:
            betas.append(num_or_error(self.betas, trait_name, keys.BETA))
            sigmas.append(num_or_error(self.vars, trait_name, keys.VAR) ** 0.5)
        return Params(
            trait_names=trait_names,
            endo_names=[self.endo_name],
            mus=[mu],
            taus=[tau],
            betas=[[beta] for beta in betas],
            sigmas=sigmas,
            trait_edges=[
                [0.0 for _ in range(len(trait_names))]
                for _ in range(len(trait_names))
            ],
        )


def num_or_error(nums: Dict[str, float], name: str, kind: str) -> float:
    if name not in nums:
        raise new_error(f"Missing value for {kind} of {name}.")
    return nums[name]


def import_phenet(options: ImportPhenetOptions) -> None:
    phenet_opts = read_opts_file(options.phenet_file)
    builder = ConfigBuilder()
    for config_file in phenet_opts.config_files:
        builder.read_phenet_config(config_file)
    if phenet_opts.output_file:
        builder.read_phenet_config_optional(phenet_opts.output_file)
    builder.report()
    config = builder.build_config(options, phenet_opts)
    config_string = dump_config(config)
    with open(options.config_file, "w", encoding="utf-8") as handle:
        handle.write(config_string)

    if builder.got_some_params():
        try:
            params = builder.build_params()
            write_params_to_file(params, options.params_file)
        except Exception as exc:
            print(f"Warning: no parameters written: {exc}")
    else:
        print("No parameter file written, because no params in phenet files given.")


def read_opts_file(opts_file: str) -> PhenetOpts:
    var_id_file = None
    config_files: list[str] = []
    output_file = None
    with open(opts_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise new_error(f"Expected key/value pair, but got '{line}'.")
            key, value = parts
            if key == keys.VAR_ID_FILE:
                var_id_file = value
            elif key == keys.CONFIG_FILE:
                config_files.append(value)
            elif key == keys.OUTPUT_FILE:
                output_file = value
            else:
                print(f"Ignoring option: {key}: {value}")
    if var_id_file is None:
        print(f"Warning: no variant id file ({keys.VAR_ID_FILE}) specified.")
        var_id_file = ""
    if not config_files:
        raise new_error(f"Missing option {keys.CONFIG_FILE}")
    return PhenetOpts(var_id_file=var_id_file, config_files=config_files, output_file=output_file)

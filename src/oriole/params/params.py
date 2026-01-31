from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterator

from ..error import new_error, MocasaError, ErrorKind, for_file


@dataclass(frozen=True)
class ParamIndex:
    kind: str
    i_trait: int | None = None
    i_endo: int | None = None

    @staticmethod
    def all(n_traits: int, n_endos: int) -> Iterator["ParamIndex"]:
        for i_endo in range(n_endos):
            yield ParamIndex("mu", i_endo=i_endo)
        for i_endo in range(n_endos):
            yield ParamIndex("tau", i_endo=i_endo)
        for i_trait in range(n_traits):
            for i_endo in range(n_endos):
                yield ParamIndex("beta", i_trait=i_trait, i_endo=i_endo)
        for i_trait in range(n_traits):
            yield ParamIndex("sigma", i_trait)

    @staticmethod
    def n_params(n_traits: int, n_endos: int) -> int:
        return (2 * n_endos) + (n_traits * n_endos) + n_traits

    def get_ordinal(self, n_traits: int, n_endos: int) -> int:
        if self.kind == "mu":
            return int(self.i_endo)
        if self.kind == "tau":
            return n_endos + int(self.i_endo)
        if self.kind == "beta":
            return (2 * n_endos) + (int(self.i_trait) * n_endos) + int(self.i_endo)
        if self.kind == "sigma":
            return (2 * n_endos) + (n_traits * n_endos) + int(self.i_trait)
        raise new_error(f"Unknown param index {self.kind}")

    def with_trait_name(self, trait_names: list[str], endo_names: list[str]) -> str:
        if self.kind == "mu":
            return f"mu_{endo_names[int(self.i_endo)]}"
        if self.kind == "tau":
            return f"tau_{endo_names[int(self.i_endo)]}"
        if self.kind == "beta":
            return f"beta_{trait_names[int(self.i_trait)]}_{endo_names[int(self.i_endo)]}"
        if self.kind == "sigma":
            return f"sigma_{trait_names[int(self.i_trait)]}"
        raise new_error(f"Unknown param index {self.kind}")

    def __str__(self) -> str:
        if self.kind in ("mu", "tau"):
            return f"{self.kind}_{self.i_endo}"
        if self.kind == "beta":
            return f"{self.kind}_{self.i_trait}_{self.i_endo}"
        return f"{self.kind}_{self.i_trait}"


@dataclass
class ParamsOverride:
    mu: float | None = None
    tau: float | None = None
    mus: list[float] | None = None
    taus: list[float] | None = None
    mus_by_name: dict[str, float] | None = None
    taus_by_name: dict[str, float] | None = None


@dataclass
class Params:
    trait_names: list[str]
    endo_names: list[str]
    mus: list[float]
    taus: list[float]
    betas: list[list[float]]
    sigmas: list[float]
    trait_edges: list[list[float]]

    @classmethod
    def from_vec(
        cls, values: list[float], trait_names: list[str], endo_names: list[str]
    ) -> "Params":
        n_traits = len(trait_names)
        n_endos = len(endo_names)
        n_values_needed = ParamIndex.n_params(n_traits, n_endos)
        if len(values) != n_values_needed:
            raise new_error(
                "Need {} values for {} traits and {} endos, but got {}.".format(
                    n_values_needed, n_traits, n_endos, len(values)
                )
            )
        cursor = 0
        mus = values[cursor : cursor + n_endos]
        cursor += n_endos
        taus = values[cursor : cursor + n_endos]
        cursor += n_endos
        betas_flat = values[cursor : cursor + (n_traits * n_endos)]
        cursor += n_traits * n_endos
        sigmas = values[cursor : cursor + n_traits]
        betas = [
            betas_flat[i_trait * n_endos : (i_trait + 1) * n_endos]
            for i_trait in range(n_traits)
        ]
        return cls(
            trait_names=trait_names,
            endo_names=endo_names,
            mus=mus,
            taus=taus,
            betas=betas,
            sigmas=sigmas,
            trait_edges=[
                [0.0 for _ in range(n_traits)] for _ in range(n_traits)
            ],
        )

    def n_traits(self) -> int:
        return len(self.trait_names)

    def n_endos(self) -> int:
        return len(self.endo_names)

    def reduce_to(self, trait_names: list[str], is_cols: list[int]) -> "Params":
        betas = [self.betas[i_col] for i_col in is_cols]
        sigmas = [self.sigmas[i_col] for i_col in is_cols]
        trait_edges = [
            [self.trait_edges[i_child][i_parent] for i_parent in is_cols]
            for i_child in is_cols
        ]
        return Params(
            trait_names=trait_names,
            endo_names=self.endo_names,
            mus=self.mus,
            taus=self.taus,
            betas=betas,
            sigmas=sigmas,
            trait_edges=trait_edges,
        )

    def plus_overwrite(self, overwrite: ParamsOverride) -> "Params":
        mus = list(self.mus)
        taus = list(self.taus)
        if overwrite.mu is not None:
            if self.n_endos() != 1:
                raise new_error("mu override only supported for a single endophenotype.")
            mus[0] = overwrite.mu
        if overwrite.tau is not None:
            if self.n_endos() != 1:
                raise new_error("tau override only supported for a single endophenotype.")
            taus[0] = overwrite.tau
        if overwrite.mus is not None:
            if len(overwrite.mus) != self.n_endos():
                raise new_error("mus override length does not match endophenotypes.")
            mus = list(overwrite.mus)
        if overwrite.taus is not None:
            if len(overwrite.taus) != self.n_endos():
                raise new_error("taus override length does not match endophenotypes.")
            taus = list(overwrite.taus)
        if overwrite.mus_by_name:
            for name, value in overwrite.mus_by_name.items():
                if name not in self.endo_names:
                    raise new_error(f"Unknown endophenotype name in mus override: {name}")
                mus[self.endo_names.index(name)] = value
        if overwrite.taus_by_name:
            for name, value in overwrite.taus_by_name.items():
                if name not in self.endo_names:
                    raise new_error(f"Unknown endophenotype name in taus override: {name}")
                taus[self.endo_names.index(name)] = value
        return Params(
            trait_names=self.trait_names,
            endo_names=self.endo_names,
            mus=mus,
            taus=taus,
            betas=self.betas,
            sigmas=self.sigmas,
            trait_edges=self.trait_edges,
        )

    def normalized_with_mu_one(self) -> "Params":
        mus = []
        taus = list(self.taus)
        betas = [list(row) for row in self.betas]
        for i_endo, mu in enumerate(self.mus):
            if mu == 0.0:
                raise new_error("Cannot normalize with mu=0.")
            mus.append(1.0)
            taus[i_endo] = taus[i_endo] / mu
            for i_trait in range(len(betas)):
                betas[i_trait][i_endo] = betas[i_trait][i_endo] * mu
        return Params(
            trait_names=self.trait_names,
            endo_names=self.endo_names,
            mus=mus,
            taus=taus,
            betas=betas,
            sigmas=self.sigmas,
            trait_edges=self.trait_edges,
        )

    def __getitem__(self, index: ParamIndex) -> float:
        if index.kind == "mu":
            return self.mus[int(index.i_endo)]
        if index.kind == "tau":
            return self.taus[int(index.i_endo)]
        if index.kind == "beta":
            return self.betas[int(index.i_trait)][int(index.i_endo)]
        if index.kind == "sigma":
            return self.sigmas[int(index.i_trait)]
        raise new_error(f"Unknown param index {index.kind}")

    def __str__(self) -> str:
        lines = []
        for name, mu, tau in zip(self.endo_names, self.mus, self.taus):
            lines.append(f"mu_{name} = {mu}")
            lines.append(f"tau_{name} = {tau}")
        for i_trait, name in enumerate(self.trait_names):
            for i_endo, endo in enumerate(self.endo_names):
                lines.append(f"beta_{name}_{endo} = {self.betas[i_trait][i_endo]}")
            lines.append(f"sigma_{name} = {self.sigmas[i_trait]}")
        return "\n".join(lines)


def read_params_from_file(path: str) -> Params:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise for_file(path, exc) from exc

    if "endo_names" in data:
        trait_names = data["trait_names"]
        endo_names = data["endo_names"]
        mus = data["mus"]
        taus = data["taus"]
        betas = data["betas"]
        sigmas = data["sigmas"]
        trait_edges = data.get("trait_edges")
        if trait_edges is None:
            print("Note: trait_edges not provided; defaulting to all zeros.")
            trait_edges = [[0.0 for _ in trait_names] for _ in trait_names]
        if len(betas) != len(trait_names):
            raise new_error("Params betas must have one row per trait.")
        for row in betas:
            if len(row) != len(endo_names):
                raise new_error("Params betas rows must match number of endophenotypes.")
        if len(trait_edges) != len(trait_names):
            raise new_error("Params trait_edges must have one row per trait.")
        for row in trait_edges:
            if len(row) != len(trait_names):
                raise new_error("Params trait_edges rows must match number of traits.")
        return Params(
            trait_names=trait_names,
            endo_names=endo_names,
            mus=mus,
            taus=taus,
            betas=betas,
            sigmas=sigmas,
            trait_edges=trait_edges,
        )
    betas = [[beta] for beta in data["betas"]]
    if len(betas) != len(data["trait_names"]):
        raise new_error("Params betas must have one row per trait.")
    print("Note: using single-endophenotype params format (endo name = 'E').")
    return Params(
        trait_names=data["trait_names"],
        endo_names=["E"],
        mus=[data["mu"]],
        taus=[data["tau"]],
        betas=betas,
        sigmas=data["sigmas"],
        trait_edges=[[0.0 for _ in data["trait_names"]] for _ in data["trait_names"]],
    )


def write_params_to_file(params: Params, output_file: str) -> None:
    data = {
        "trait_names": params.trait_names,
        "endo_names": params.endo_names,
        "mus": params.mus,
        "taus": params.taus,
        "betas": params.betas,
        "sigmas": params.sigmas,
        "trait_edges": params.trait_edges,
    }
    try:
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
            handle.write("\n")
    except Exception as exc:
        raise for_file(output_file, exc) from exc

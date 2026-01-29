from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterator

from ..error import new_error, MocasaError, ErrorKind, for_file


@dataclass(frozen=True)
class ParamIndex:
    kind: str
    i_trait: int | None = None

    @staticmethod
    def all(n_traits: int) -> Iterator["ParamIndex"]:
        yield ParamIndex("mu")
        yield ParamIndex("tau")
        for i_trait in range(n_traits):
            yield ParamIndex("beta", i_trait)
        for i_trait in range(n_traits):
            yield ParamIndex("sigma", i_trait)

    @staticmethod
    def n_params(n_traits: int) -> int:
        return 2 * n_traits + 2

    def get_ordinal(self, n_traits: int) -> int:
        if self.kind == "mu":
            return 0
        if self.kind == "tau":
            return 1
        if self.kind == "beta":
            return 2 + int(self.i_trait)
        if self.kind == "sigma":
            return 2 + n_traits + int(self.i_trait)
        raise new_error(f"Unknown param index {self.kind}")

    def with_trait_name(self, trait_names: list[str]) -> str:
        if self.kind == "mu":
            return "mu"
        if self.kind == "tau":
            return "tau"
        if self.kind == "beta":
            return f"beta_{trait_names[int(self.i_trait)]}"
        if self.kind == "sigma":
            return f"sigma_{trait_names[int(self.i_trait)]}"
        raise new_error(f"Unknown param index {self.kind}")

    def __str__(self) -> str:
        if self.kind in ("mu", "tau"):
            return self.kind
        return f"{self.kind}_{self.i_trait}"


@dataclass
class ParamsOverride:
    mu: float | None = None
    tau: float | None = None


@dataclass
class Params:
    trait_names: list[str]
    mu: float
    tau: float
    betas: list[float]
    sigmas: list[float]

    @classmethod
    def from_vec(cls, values: list[float], trait_names: list[str]) -> "Params":
        n_traits = len(trait_names)
        n_values_needed = ParamIndex.n_params(n_traits)
        if len(values) != n_values_needed:
            raise new_error(
                f"Need {n_values_needed} values for {n_traits} traits, but got {len(values)}."
            )
        mu = values[0]
        tau = values[1]
        betas = values[2 : 2 + n_traits]
        sigmas = values[2 + n_traits : 2 + 2 * n_traits]
        return cls(trait_names=trait_names, mu=mu, tau=tau, betas=betas, sigmas=sigmas)

    def n_traits(self) -> int:
        return len(self.trait_names)

    def reduce_to(self, trait_names: list[str], is_cols: list[int]) -> "Params":
        betas = [self.betas[i_col] for i_col in is_cols]
        sigmas = [self.sigmas[i_col] for i_col in is_cols]
        return Params(trait_names=trait_names, mu=self.mu, tau=self.tau, betas=betas, sigmas=sigmas)

    def plus_overwrite(self, overwrite: ParamsOverride) -> "Params":
        mu = overwrite.mu if overwrite.mu is not None else self.mu
        tau = overwrite.tau if overwrite.tau is not None else self.tau
        return Params(
            trait_names=self.trait_names,
            mu=mu,
            tau=tau,
            betas=self.betas,
            sigmas=self.sigmas,
        )

    def normalized_with_mu_one(self) -> "Params":
        mu = 1.0
        tau = self.tau / self.mu
        betas = [beta * self.mu for beta in self.betas]
        return Params(
            trait_names=self.trait_names,
            mu=mu,
            tau=tau,
            betas=betas,
            sigmas=self.sigmas,
        )

    def __getitem__(self, index: ParamIndex) -> float:
        if index.kind == "mu":
            return self.mu
        if index.kind == "tau":
            return self.tau
        if index.kind == "beta":
            return self.betas[int(index.i_trait)]
        if index.kind == "sigma":
            return self.sigmas[int(index.i_trait)]
        raise new_error(f"Unknown param index {index.kind}")

    def __str__(self) -> str:
        lines = [f"mu = {self.mu}"]
        for name, beta, sigma in zip(self.trait_names, self.betas, self.sigmas):
            lines.append(f"beta_{name} = {beta}")
            lines.append(f"sigma_{name} = {sigma}")
        return "\n".join(lines)


def read_params_from_file(path: str) -> Params:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise for_file(path, exc) from exc

    return Params(
        trait_names=data["trait_names"],
        mu=data["mu"],
        tau=data["tau"],
        betas=data["betas"],
        sigmas=data["sigmas"],
    )


def write_params_to_file(params: Params, output_file: str) -> None:
    data = {
        "trait_names": params.trait_names,
        "mu": params.mu,
        "tau": params.tau,
        "betas": params.betas,
        "sigmas": params.sigmas,
    }
    try:
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
            handle.write("\n")
    except Exception as exc:
        raise for_file(output_file, exc) from exc

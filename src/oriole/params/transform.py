from __future__ import annotations

from ..options.cli import ScaleSigmasOptions
from .params import read_params_from_file, write_params_to_file, Params


def scale_sigmas(config: ScaleSigmasOptions) -> None:
    params = read_params_from_file(config.in_file)
    sigmas = [sigma * config.scale for sigma in params.sigmas]
    scaled = Params(
        trait_names=params.trait_names,
        endo_names=params.endo_names,
        mus=params.mus,
        taus=params.taus,
        betas=params.betas,
        sigmas=sigmas,
        trait_edges=params.trait_edges,
    )
    write_params_to_file(scaled, config.out_file)

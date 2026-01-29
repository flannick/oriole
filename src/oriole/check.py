from __future__ import annotations

from .error import new_error
from .options.config import Config
from .params import Params


def check_config(config: Config) -> None:
    if not config.gwas:
        raise new_error("No GWAS specified.")


def check_params(config: Config, params: Params) -> None:
    if len(config.gwas) != len(params.trait_names):
        raise new_error(
            "Number GWAS files ({}) does not match number of traits in params ({})".format(
                len(config.gwas), len(params.trait_names)
            )
        )
    for i_trait, (gwas, trait_name) in enumerate(
        zip(config.gwas, params.trait_names)
    ):
        if gwas.name != trait_name:
            raise new_error(
                "Trait name in GWAS file {} ({}) does not match trait name in params ({})".format(
                    i_trait, gwas.name, trait_name
                )
            )


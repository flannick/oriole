from __future__ import annotations

from .check import check_config
from .error import MocasaError, new_error
from .options.cli import get_choice, CoreOptions, ImportPhenetOptions, ScaleSigmasOptions
from .options.config import load_config
from .options.check_pre import check_prerequisites
from .options.action import Action
from .train.train import train_or_check
from .classify.classify import classify_or_check
from .phenet import import_phenet
from .params.transform import scale_sigmas


def run(argv: list[str] | None = None) -> None:
    choice = get_choice(argv)
    if isinstance(choice, CoreOptions):
        config = load_config(choice.config_file)
        check_config(config)
        check_prerequisites(config)
        if choice.plot_convergence_out_file:
            config.train.plot_convergence_out_file = choice.plot_convergence_out_file
        if choice.plot_cv_out_file:
            config.train.plot_cv_out_file = choice.plot_cv_out_file
        if choice.action == Action.TRAIN:
            train_or_check(
                config,
                choice.dry,
                match_rust=choice.match_rust,
                inference=choice.inference,
                chunk_size=choice.chunk_size,
                verbose=choice.verbose,
            )
        else:
            classify_or_check(
                config,
                choice.dry,
                inference=choice.inference,
                chunk_size=choice.chunk_size,
                verbose=choice.verbose,
            )
    elif isinstance(choice, ImportPhenetOptions):
        import_phenet(choice)
    elif isinstance(choice, ScaleSigmasOptions):
        scale_sigmas(choice)
    else:
        raise MocasaError("Mocasa error", "Unknown choice")


def main(argv: list[str] | None = None) -> None:
    try:
        run(argv)
        print("Done!")
    except MocasaError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

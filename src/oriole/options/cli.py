from __future__ import annotations

import argparse
from dataclasses import dataclass

from ..error import new_error, MocasaError
from .action import Action


@dataclass
class CoreOptions:
    action: Action
    config_file: str
    dry: bool
    match_rust: bool
    analytical: bool
    chunk_size: int | None


@dataclass
class ImportPhenetOptions:
    phenet_file: str
    params_file: str
    config_file: str
    out_file: str


@dataclass
class ScaleSigmasOptions:
    in_file: str
    scale: float
    out_file: str


def _missing_option_error(name: str, long_opt: str, short_opt: str) -> MocasaError:
    return new_error(f"Missing {name} option ('--{long_opt}' or '-{short_opt}').")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oriole")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_core(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("-f", "--conf-file", dest="conf_file")
        sub.add_argument("-d", "--dry", action="store_true")
        sub.add_argument("--match-rust", action="store_true", dest="match_rust")
        sub.add_argument("--analytical", action="store_true", dest="analytical")
        sub.add_argument(
            "--chunk-size",
            dest="chunk_size",
            type=int,
            help="Number of variants to process per chunk (default ~2GB).",
        )

    train = subparsers.add_parser(Action.TRAIN.value)
    add_core(train)

    classify = subparsers.add_parser(Action.CLASSIFY.value)
    add_core(classify)

    import_phenet = subparsers.add_parser("import-phenet")
    import_phenet.add_argument("-i", "--phenet-file", dest="phenet_file")
    import_phenet.add_argument("-p", "--params-file", dest="params_file")
    import_phenet.add_argument("-f", "--conf-file", dest="conf_file")
    import_phenet.add_argument("-o", "--out-file", dest="out_file")

    scale_sigmas = subparsers.add_parser("scale-sigmas")
    scale_sigmas.add_argument("-i", "--in-file", dest="in_file")
    scale_sigmas.add_argument("-s", "--scale", dest="scale", type=float)
    scale_sigmas.add_argument("-o", "--out-file", dest="out_file")

    return parser


def get_choice(argv: list[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in (Action.TRAIN.value, Action.CLASSIFY.value):
        if not args.conf_file:
            raise _missing_option_error("config file", "conf-file", "f")
        action = Action(args.command)
        return CoreOptions(
            action=action,
            config_file=args.conf_file,
            dry=bool(args.dry),
            match_rust=bool(args.match_rust),
            analytical=bool(args.analytical),
            chunk_size=args.chunk_size,
        )

    if args.command == "import-phenet":
        if not args.phenet_file:
            raise _missing_option_error("phenet opts file", "phenet-file", "i")
        if not args.params_file:
            raise _missing_option_error("Mocasa parameters file", "params-file", "p")
        if not args.conf_file:
            raise _missing_option_error("Mocasa config file", "conf-file", "f")
        if not args.out_file:
            raise _missing_option_error("Mocasa classification output file", "out-file", "o")
        return ImportPhenetOptions(
            phenet_file=args.phenet_file,
            params_file=args.params_file,
            config_file=args.conf_file,
            out_file=args.out_file,
        )

    if args.command == "scale-sigmas":
        if not args.in_file:
            raise _missing_option_error("input params file", "in-file", "i")
        if args.scale is None:
            raise _missing_option_error("scale", "scale", "s")
        if not args.out_file:
            raise _missing_option_error("output params file", "out-file", "o")
        return ScaleSigmasOptions(in_file=args.in_file, scale=args.scale, out_file=args.out_file)

    raise new_error(f"Unknown subcommand {args.command}.")

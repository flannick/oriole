from __future__ import annotations

from pathlib import Path

from ..params import ParamIndex, Params


class ParamTraceFileWriter:
    def __init__(self, path: Path, n_traits: int) -> None:
        self.path = path
        self.index = 0
        with self.path.open("w", encoding="utf-8") as handle:
            handle.write("index")
            for param_index in ParamIndex.all(n_traits):
                handle.write(f"\t{param_index}")
            handle.write("\n")

    def write(self, params: Params) -> None:
        self.index += 1
        n_traits = params.n_traits()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(str(self.index))
            for param_index in ParamIndex.all(n_traits):
                handle.write(f"\t{params[param_index]}")
            handle.write("\n")

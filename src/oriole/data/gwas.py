from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from ..error import new_error


@dataclass
class GwasCols:
    id: str
    effect: str
    se: str


class default_cols:
    VAR_ID = "VAR_ID"
    BETA = "BETA"
    SE = "SE"


def _get_delim(header: str) -> str:
    for delim in [";", "\t", ",", " "]:
        if delim in header:
            return delim
    raise new_error(
        "Can only parse data files with semicolon, tab, comma or single blank as delimiter."
    )


@dataclass
class GwasRecord:
    var_id: str
    beta: float
    se: float


class GwasReader(Iterator[GwasRecord]):
    def __init__(self, lines: Iterable[str], cols: GwasCols) -> None:
        self._lines_iter = iter(lines)
        try:
            header = next(self._lines_iter).rstrip("\n")
        except StopIteration as exc:
            raise new_error("File is empty") from exc
        self._delim = _get_delim(header)
        self._cols = cols
        header_parts = header.split(self._delim)
        self._i_var_id = self._index_of(header_parts, cols.id)
        self._i_beta = self._index_of(header_parts, cols.effect)
        self._i_se = self._index_of(header_parts, cols.se)

    @staticmethod
    def _index_of(parts: list[str], name: str) -> int:
        try:
            return parts.index(name)
        except ValueError as exc:
            raise new_error(
                "No {} column. Available columns: {}".format(name, ", ".join(parts))
            ) from exc

    def __next__(self) -> GwasRecord:
        line = next(self._lines_iter)
        return self.parse_line(line.rstrip("\n"))

    def parse_line(self, line: str) -> GwasRecord:
        var_id = None
        beta = None
        se = None
        for i, part in enumerate(line.split(self._delim)):
            if i == self._i_var_id:
                var_id = part
            elif i == self._i_beta:
                beta = float(part)
            elif i == self._i_se:
                se = float(part)
            if var_id is not None and beta is not None and se is not None:
                break
        if var_id is None:
            raise new_error(f"Missing value for '{self._cols.id}'.")
        if beta is None:
            raise new_error(f"Missing value for '{self._cols.effect}'.")
        if se is None:
            raise new_error(f"Missing value for '{self._cols.se}'.")
        return GwasRecord(var_id=var_id, beta=beta, se=se)


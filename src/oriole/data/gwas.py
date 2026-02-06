from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from ..error import new_error


@dataclass
class GwasCols:
    id: str
    effect: str
    se: str
    chrom: str | None = None
    pos: str | None = None
    effect_allele: str | None = None
    other_allele: str | None = None
    eaf: str | None = None
    rsid: str | None = None


class default_cols:
    VAR_ID = "VAR_ID"
    BETA = "BETA"
    SE = "SE"


META_COL_CANDIDATES = {
    "chrom": ["CHR", "CHROM", "CHROMOSOME", "#CHROM"],
    "pos": ["BP", "POS", "POSITION", "BASE_PAIR_LOCATION"],
    "effect_allele": ["EA", "EFFECT_ALLELE", "A1", "ALLELE1", "ALT", "ALT_ALLELE"],
    "other_allele": ["NEA", "OTHER_ALLELE", "A2", "ALLELE2", "REF", "REF_ALLELE"],
    "eaf": ["EAF", "EFFECT_ALLELE_FREQUENCY", "FRQ", "FREQ", "AF", "A1FREQ"],
    "rsid": ["RSID", "RS_ID", "SNP", "ID"],
}


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
    chrom: str | None = None
    pos: int | None = None
    effect_allele: str | None = None
    other_allele: str | None = None
    eaf: float | None = None
    rsid: str | None = None


class GwasReader(Iterator[GwasRecord]):
    def __init__(self, lines: Iterable[str], cols: GwasCols, auto_meta: bool = True) -> None:
        self._lines_iter = iter(lines)
        try:
            header = next(self._lines_iter).rstrip("\n")
        except StopIteration as exc:
            raise new_error("File is empty") from exc
        self._delim = _get_delim(header)
        self._cols = cols
        self._auto_meta = auto_meta
        header_parts = header.split(self._delim)
        self._i_var_id = self._index_of(header_parts, cols.id)
        self._i_beta = self._index_of(header_parts, cols.effect)
        self._i_se = self._index_of(header_parts, cols.se)
        self._i_chr = self._index_of_optional(
            header_parts, cols.chrom, META_COL_CANDIDATES["chrom"]
        )
        self._i_pos = self._index_of_optional(
            header_parts, cols.pos, META_COL_CANDIDATES["pos"]
        )
        self._i_effect_allele = self._index_of_optional(
            header_parts, cols.effect_allele, META_COL_CANDIDATES["effect_allele"]
        )
        self._i_other_allele = self._index_of_optional(
            header_parts, cols.other_allele, META_COL_CANDIDATES["other_allele"]
        )
        self._i_eaf = self._index_of_optional(
            header_parts, cols.eaf, META_COL_CANDIDATES["eaf"]
        )
        self._i_rsid = self._index_of_optional(
            header_parts, cols.rsid, META_COL_CANDIDATES["rsid"]
        )

    @staticmethod
    def _index_of(parts: list[str], name: str) -> int:
        try:
            return parts.index(name)
        except ValueError as exc:
            raise new_error(
                "No {} column. Available columns: {}".format(name, ", ".join(parts))
            ) from exc

    def _index_of_optional(
        self, parts: list[str], name: str | None, candidates: list[str]
    ) -> int | None:
        if name:
            return self._index_of(parts, name)
        if not self._auto_meta:
            return None
        for candidate in candidates:
            if candidate in parts:
                return parts.index(candidate)
        return None

    def __next__(self) -> GwasRecord:
        line = next(self._lines_iter)
        return self.parse_line(line.rstrip("\n"))

    def parse_line(self, line: str) -> GwasRecord:
        var_id = None
        beta = None
        se = None
        chrom = None
        pos = None
        effect_allele = None
        other_allele = None
        eaf = None
        rsid = None
        needs_meta = any(
            idx is not None
            for idx in (
                self._i_chr,
                self._i_pos,
                self._i_effect_allele,
                self._i_other_allele,
                self._i_eaf,
                self._i_rsid,
            )
        )
        for i, part in enumerate(line.split(self._delim)):
            if i == self._i_var_id:
                var_id = part
            elif i == self._i_beta:
                beta = float(part)
            elif i == self._i_se:
                se = float(part)
            elif self._i_chr is not None and i == self._i_chr:
                chrom = part or None
            elif self._i_pos is not None and i == self._i_pos:
                if part and part.upper() != "NA":
                    pos = int(part)
            elif self._i_effect_allele is not None and i == self._i_effect_allele:
                effect_allele = part or None
            elif self._i_other_allele is not None and i == self._i_other_allele:
                other_allele = part or None
            elif self._i_eaf is not None and i == self._i_eaf:
                if part and part.upper() != "NA":
                    eaf = float(part)
            elif self._i_rsid is not None and i == self._i_rsid:
                rsid = part or None
            if var_id is not None and beta is not None and se is not None and not needs_meta:
                break
        if var_id is None:
            raise new_error(f"Missing value for '{self._cols.id}'.")
        if beta is None:
            raise new_error(f"Missing value for '{self._cols.effect}'.")
        if se is None:
            raise new_error(f"Missing value for '{self._cols.se}'.")
        return GwasRecord(
            var_id=var_id,
            beta=beta,
            se=se,
            chrom=chrom,
            pos=pos,
            effect_allele=effect_allele,
            other_allele=other_allele,
            eaf=eaf,
            rsid=rsid,
        )

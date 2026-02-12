from __future__ import annotations

import csv
import gzip
import io
import math
import re
from dataclasses import dataclass

from ..error import new_error

_VAR_ID_GUESS = re.compile(
    r"^(?:chr)?(?P<chrom>[A-Za-z0-9]+)[^A-Za-z0-9]+(?P<pos>\d+)[^A-Za-z0-9]+(?P<a1>[A-Za-z]+)[^A-Za-z0-9]+(?P<a2>[A-Za-z]+)$",
    re.IGNORECASE,
)


@dataclass
class VariantMetaGuess:
    chrom: str | None
    pos: int | None
    effect_allele: str | None
    other_allele: str | None


def _open_text(path: str, mode: str = "r"):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return io.open(path, mode, encoding="utf-8")


def _guess_variant_meta(var_id: str, variant_id_order: str) -> VariantMetaGuess:
    match = _VAR_ID_GUESS.match(var_id)
    if match is None:
        return VariantMetaGuess(None, None, None, None)
    chrom = match.group("chrom")
    pos = int(match.group("pos"))
    a1 = match.group("a1")
    a2 = match.group("a2")
    if variant_id_order == "other_effect":
        a1, a2 = a2, a1
    return VariantMetaGuess(
        chrom=chrom,
        pos=pos,
        effect_allele=a1,
        other_allele=a2,
    )


def _endo_names_from_header(fieldnames: list[str]) -> list[str]:
    names: list[str] = []
    seen = set()
    for name in fieldnames:
        if name.endswith("_beta_gls"):
            endo = name[: -len("_beta_gls")]
            if endo not in seen:
                seen.add(endo)
                names.append(endo)
    return names


def _ssf_out_paths(out_file: str, endo_names: list[str]) -> dict[str, str]:
    if len(endo_names) == 1:
        return {endo_names[0]: out_file}
    gz = out_file.endswith(".gz")
    base = out_file[:-3] if gz else out_file
    stem, ext = base.rsplit(".", 1) if "." in base else (base, "")
    out: dict[str, str] = {}
    for endo in endo_names:
        path = f"{stem}_{endo}"
        if ext:
            path = f"{path}.{ext}"
        if gz:
            path = f"{path}.gz"
        out[endo] = path
    return out


def convert_classify_tsv_to_ssf(
    in_file: str,
    out_file: str,
    guess_fields: bool = True,
    variant_id_order: str = "effect_other",
) -> None:
    if variant_id_order not in {"effect_other", "other_effect"}:
        raise new_error(
            "variant_id_order must be 'effect_other' or 'other_effect'."
        )

    with _open_text(in_file, "r") as in_handle:
        reader = csv.DictReader(in_handle, delimiter="\t")
        if not reader.fieldnames:
            raise new_error(f"Input file has no header: {in_file}")
        if "id" not in reader.fieldnames:
            raise new_error(
                "Input file must be a full ORIOLE classify output with an 'id' column."
            )

        endo_names = _endo_names_from_header(reader.fieldnames)
        if not endo_names:
            raise new_error(
                "Input file has no '*_beta_gls' columns; cannot build GWAS-SSF output."
            )

        out_paths = _ssf_out_paths(out_file, endo_names)
        handles = {
            endo: _open_text(path, "w") for endo, path in out_paths.items()
        }
        try:
            header = ["variant_id"]
            if guess_fields:
                header.extend(
                    [
                        "chromosome",
                        "base_pair_location",
                        "effect_allele",
                        "other_allele",
                    ]
                )
            header.extend(["beta", "standard_error", "p_value"])
            header_line = "\t".join(header) + "\n"
            for handle in handles.values():
                handle.write(header_line)

            for row in reader:
                var_id = row["id"]
                guessed = (
                    _guess_variant_meta(var_id, variant_id_order)
                    if guess_fields
                    else VariantMetaGuess(None, None, None, None)
                )
                for endo in endo_names:
                    beta_key = f"{endo}_beta_gls"
                    se_key = f"{endo}_se_gls"
                    p_key = f"{endo}_p_gls"
                    z_key = f"{endo}_z_gls"
                    if beta_key not in row or se_key not in row:
                        raise new_error(
                            f"Missing required columns for endophenotype '{endo}': "
                            f"{beta_key} and/or {se_key}."
                        )
                    beta = row[beta_key]
                    se = row[se_key]
                    p_val = row.get(p_key, "")
                    if p_val == "":
                        z_val = row.get(z_key, "")
                        try:
                            z_float = float(z_val)
                            p_val = str(math.erfc(abs(z_float) / math.sqrt(2.0)))
                        except Exception:
                            p_val = ""

                    parts = [var_id]
                    if guess_fields:
                        parts.extend(
                            [
                                "" if guessed.chrom is None else str(guessed.chrom),
                                "" if guessed.pos is None else str(guessed.pos),
                                ""
                                if guessed.effect_allele is None
                                else guessed.effect_allele,
                                ""
                                if guessed.other_allele is None
                                else guessed.other_allele,
                            ]
                        )
                    parts.extend([beta, se, p_val])
                    handles[endo].write("\t".join(parts) + "\n")
        finally:
            for handle in handles.values():
                handle.close()

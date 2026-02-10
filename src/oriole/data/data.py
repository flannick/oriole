from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import time
import gzip
import io
import hashlib

import numpy as np
import re
import math

from ..error import new_error, for_context, for_file
from ..math_utils.matrix import matrix_fill
from ..options.action import Action
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..options.config import Config, GwasConfig, DataAccessConfig
from .gwas import GwasReader, GwasRecord, GwasCols, default_cols

try:
    from dig_open_data import (
        open_text as open_data_text,
        build_key as dig_build_key,
        DEFAULT_BUCKET as DIG_DEFAULT_BUCKET,
        DEFAULT_PREFIX as DIG_DEFAULT_PREFIX,
        DEFAULT_SUFFIX as DIG_DEFAULT_SUFFIX,
        CacheConfig as DigCacheConfig,
    )
except Exception:  # pragma: no cover - optional dependency
    open_data_text = None
    dig_build_key = None
    DIG_DEFAULT_BUCKET = None
    DIG_DEFAULT_PREFIX = None
    DIG_DEFAULT_SUFFIX = None
    DigCacheConfig = None

DELIM_LIST = [";", "\t", ",", " "]
_LOCUS_HEADERS = {
    "chrom": {"CHR", "CHROM", "CHROMOSOME", "#CHROM"},
    "pos": {"BP", "POS", "POSITION", "BASE_PAIR_LOCATION"},
    "ref": {"REF", "REF_ALLELE"},
    "alt": {"ALT", "ALT_ALLELE"},
    "weight": {"WEIGHT", "WEIGHTS"},
}


@dataclass
class FlipStats:
    matched: int = 0
    flipped: int = 0
    missing_meta: int = 0
    missing_id: int = 0


def _detect_delim(line: str) -> str:
    for delim in DELIM_LIST:
        if delim in line:
            return delim
    return " "


def _normalize_chrom(chrom: str) -> str:
    chrom = chrom.strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom


def _normalize_allele(allele: str) -> str:
    return allele.strip().upper()


def _locus_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{chrom}:{pos}:{ref}:{alt}"


def _bucket_for_key(key: str, n_buckets: int) -> int:
    digest = hashlib.md5(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return value % n_buckets


@dataclass
class Meta:
    trait_names: list[str]
    var_ids: list[str]
    endo_names: list[str]

    def n_data_points(self) -> int:
        return len(self.var_ids)

    def n_traits(self) -> int:
        return len(self.trait_names)

    def n_endos(self) -> int:
        return len(self.endo_names)


@dataclass
class GwasData:
    meta: Meta
    betas: np.ndarray
    ses: np.ndarray

    def n_data_points(self) -> int:
        return self.meta.n_data_points()

    def n_traits(self) -> int:
        return self.meta.n_traits()

    def only_data_point(self, i_row: int) -> tuple["GwasData", list[int]]:
        var_id = self.meta.var_ids[i_row]
        is_col = [
            i_col
            for i_col in range(self.n_traits())
            if np.isfinite(self.betas[i_row, i_col]) and np.isfinite(self.ses[i_row, i_col])
        ]
        trait_names = [self.meta.trait_names[i_col] for i_col in is_col]
        betas = self.betas[i_row, is_col].reshape(1, -1)
        ses = self.ses[i_row, is_col].reshape(1, -1)
        meta = Meta(trait_names=trait_names, var_ids=[var_id], endo_names=self.meta.endo_names)
        return GwasData(meta=meta, betas=betas, ses=ses), is_col

    def __str__(self) -> str:
        lines = [default_cols.VAR_ID]
        for trait_name in self.meta.trait_names:
            lines.append(f"beta_{trait_name}")
            lines.append(f"se_{trait_name}")
        out = ["\t".join(lines)]
        for i_data_point, var_id in enumerate(self.meta.var_ids):
            row = [var_id]
            for i_trait in range(self.n_traits()):
                row.append(str(self.betas[i_data_point, i_trait]))
                row.append(str(self.ses[i_data_point, i_trait]))
            out.append("\t".join(row))
        return "\n".join(out)


@dataclass
class Weights:
    weights: list[float]
    sum: float

    @classmethod
    def new(cls, n_data_points: int) -> "Weights":
        return cls(weights=[], sum=0.0)

    def add(self, weight: float) -> None:
        self.weights.append(weight)
        self.sum += weight


@dataclass
class LoadedData:
    gwas_data: GwasData
    weights: Weights
    variant_meta: dict[str, "VariantMeta"] | None = None


@dataclass
class BetaSe:
    beta: float
    se: float


@dataclass
class IdData:
    beta_se_list: list[BetaSe]
    weight: float


@dataclass
class VariantMeta:
    chrom: str | None
    pos: int | None
    effect_allele: str | None
    other_allele: str | None
    rsid: str | None
    eaf: float | None


@dataclass
class VariantMetaAccumulator:
    chrom: str | None = None
    pos: int | None = None
    effect_allele: str | None = None
    other_allele: str | None = None
    rsid: str | None = None
    eaf_sum: float = 0.0
    eaf_count: int = 0

    def add_record(self, record: GwasRecord, trait_name: str, var_id: str) -> None:
        self._merge_field("chrom", record.chrom, trait_name, var_id)
        self._merge_field("pos", record.pos, trait_name, var_id)
        self._merge_field("effect_allele", record.effect_allele, trait_name, var_id)
        self._merge_field("other_allele", record.other_allele, trait_name, var_id)
        self._merge_field("rsid", record.rsid, trait_name, var_id)
        if record.eaf is not None and math.isfinite(record.eaf):
            self.eaf_sum += record.eaf
            self.eaf_count += 1

    def _merge_field(
        self,
        field: str,
        value: str | int | None,
        trait_name: str,
        var_id: str,
    ) -> None:
        if value is None or value == "":
            return
        existing = getattr(self, field)
        if existing is None:
            setattr(self, field, value)
            return
        if existing != value:
            raise new_error(
                "Conflicting {} for variant {} across traits (existing={}, new={}, trait={}).".format(
                    field, var_id, existing, value, trait_name
                )
            )

    def finalize(self) -> VariantMeta:
        eaf = None
        if self.eaf_count > 0:
            eaf = self.eaf_sum / self.eaf_count
        return VariantMeta(
            chrom=self.chrom,
            pos=self.pos,
            effect_allele=self.effect_allele,
            other_allele=self.other_allele,
            rsid=self.rsid,
            eaf=eaf,
        )


def load_data(config: Config, action: Action) -> LoadedData:
    n_traits = len(config.gwas)
    need_meta = (
        action == Action.CLASSIFY
        and config.classify.gwas_ssf_out_file is not None
    )
    allow_meta_guess = bool(config.classify.gwas_ssf_guess_fields) if need_meta else False
    guess_order = (
        config.classify.gwas_ssf_variant_id_order if need_meta else "effect_other"
    )
    meta_by_id: dict[str, VariantMetaAccumulator] | None = (
        {} if need_meta else None
    )
    variant_mode = resolve_variant_mode(config, action)
    if action == Action.TRAIN:
        print(f"Loading training IDs from {config.train.ids_file}")
        ids_start = time.perf_counter()
        beta_se_by_id = load_ids(config.train.ids_file, n_traits, variant_mode)
        print(
            f"Loaded {len(beta_se_by_id)} training IDs in "
            f"{time.perf_counter() - ids_start:.2f}s"
        )
    else:
        print("Loading full GWAS for classification (all IDs).")
        beta_se_by_id = {}

    trait_names: list[str] = []
    gwas_start = time.perf_counter()
    flip_stats: dict[str, FlipStats] = {}
    for i_trait, gwas in enumerate(config.gwas):
        trait_names.append(gwas.name)
        trait_start = time.perf_counter()
        load_gwas(
            beta_se_by_id,
            gwas,
            n_traits,
            i_trait,
            action,
            data_access=config.data_access,
            meta_by_id=meta_by_id,
            auto_meta=allow_meta_guess,
            variant_mode=variant_mode,
            flip_stats=flip_stats,
        )
        gwas_label = gwas.uri or gwas.file or ""
        print(
            f"Loaded GWAS {gwas.name} from {gwas_label} in "
            f"{time.perf_counter() - trait_start:.2f}s"
        )
    print(f"Finished GWAS load in {time.perf_counter() - gwas_start:.2f}s")
    _write_flip_log(config.files.flip_log, flip_stats)

    n_data_points = len(beta_se_by_id)
    var_ids: list[str] = []
    betas = matrix_fill(n_data_points, n_traits, lambda _i, _j: np.nan)
    ses = matrix_fill(n_data_points, n_traits, lambda _i, _j: np.nan)
    weights = Weights.new(n_data_points)

    for i_data_point, (var_id, id_data) in enumerate(sorted(beta_se_by_id.items())):
        var_ids.append(var_id)
        weights.add(id_data.weight)
        for i_trait, beta_se in enumerate(id_data.beta_se_list):
            betas[i_data_point, i_trait] = beta_se.beta
            ses[i_data_point, i_trait] = beta_se.se

    if action == Action.TRAIN:
        missing_by_trait: dict[str, list[str]] = {name: [] for name in trait_names}
        for i_data_point, var_id in enumerate(var_ids):
            for i_trait, trait_name in enumerate(trait_names):
                if np.isnan(betas[i_data_point, i_trait]) or np.isnan(
                    ses[i_data_point, i_trait]
                ):
                    if len(missing_by_trait[trait_name]) < 5:
                        missing_by_trait[trait_name].append(var_id)
        missing_counts = {
            name: len(ids)
            for name, ids in missing_by_trait.items()
            if len(ids) > 0
        }
        if missing_counts:
            parts = []
            for name, count in missing_counts.items():
                examples = ", ".join(missing_by_trait[name])
                parts.append(f"{name}: {count} missing (e.g., {examples})")
            print(
                "Warning: missing GWAS entries for training IDs; "
                "treating missing traits as unobserved. "
                + " | ".join(parts)
            )

    endo_names = [item.name for item in config.endophenotypes]
    meta = Meta(trait_names=trait_names, var_ids=var_ids, endo_names=endo_names)
    gwas_data = GwasData(meta=meta, betas=betas, ses=ses)
    variant_meta = None
    if need_meta and meta_by_id is not None:
        variant_meta = finalize_variant_meta(
            meta_by_id, var_ids, allow_meta_guess, guess_order
        )
    return LoadedData(gwas_data=gwas_data, weights=weights, variant_meta=variant_meta)


def estimate_gwas_counts(config: Config) -> dict[str, int]:
    counts: dict[str, int] = {}
    cache = _build_dig_cache(config.data_access)
    for gwas in config.gwas:
        file = _resolve_gwas_path(gwas, config.data_access)
        try:
            with _open_text(
                file,
                retries=config.data_access.retries,
                download=config.data_access.download,
                cache=cache,
            ) as handle:
                count = -1
                for count, _line in enumerate(handle):
                    pass
                counts[gwas.name] = max(0, count)
        except Exception as exc:
            raise for_context(file, exc) from exc
    return counts


def load_data_bucket(
    config: Config,
    action: Action,
    bucket_id: int,
    n_buckets: int,
    flip_stats: dict[str, FlipStats] | None = None,
) -> LoadedData:
    if action != Action.CLASSIFY:
        raise new_error("Bucketed loading is only supported for classification.")
    n_traits = len(config.gwas)
    need_meta = (
        action == Action.CLASSIFY and config.classify.gwas_ssf_out_file is not None
    )
    allow_meta_guess = bool(config.classify.gwas_ssf_guess_fields) if need_meta else False
    guess_order = (
        config.classify.gwas_ssf_variant_id_order if need_meta else "effect_other"
    )
    meta_by_id: dict[str, VariantMetaAccumulator] | None = (
        {} if need_meta else None
    )
    beta_se_by_id: Dict[str, IdData] = {}
    trait_names: list[str] = []
    for i_trait, gwas in enumerate(config.gwas):
        trait_names.append(gwas.name)
        load_gwas(
            beta_se_by_id,
            gwas,
            n_traits,
            i_trait,
            action,
            data_access=config.data_access,
            meta_by_id=meta_by_id,
            auto_meta=allow_meta_guess,
            variant_mode=config.variants.id_mode,
            flip_stats=flip_stats,
            bucket_id=bucket_id,
            n_buckets=n_buckets,
        )

    n_data_points = len(beta_se_by_id)
    var_ids: list[str] = []
    betas = matrix_fill(n_data_points, n_traits, lambda _i, _j: np.nan)
    ses = matrix_fill(n_data_points, n_traits, lambda _i, _j: np.nan)

    for i_data_point, (var_id, id_data) in enumerate(sorted(beta_se_by_id.items())):
        var_ids.append(var_id)
        for i_trait, beta_se in enumerate(id_data.beta_se_list):
            betas[i_data_point, i_trait] = beta_se.beta
            ses[i_data_point, i_trait] = beta_se.se

    endo_names = [item.name for item in config.endophenotypes]
    meta = Meta(trait_names=trait_names, var_ids=var_ids, endo_names=endo_names)
    gwas_data = GwasData(meta=meta, betas=betas, ses=ses)
    variant_meta = None
    if need_meta and meta_by_id is not None:
        variant_meta = finalize_variant_meta(
            meta_by_id, var_ids, allow_meta_guess, guess_order
        )
    weights = Weights.new(n_data_points)
    return LoadedData(gwas_data=gwas_data, weights=weights, variant_meta=variant_meta)


def _parse_ids_header(parts: list[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    upper = [p.strip().upper() for p in parts]
    for i, value in enumerate(upper):
        if value in _LOCUS_HEADERS["chrom"]:
            indices["chrom"] = i
        elif value in _LOCUS_HEADERS["pos"]:
            indices["pos"] = i
        elif value in _LOCUS_HEADERS["ref"]:
            indices["ref"] = i
        elif value in _LOCUS_HEADERS["alt"]:
            indices["alt"] = i
        elif value in _LOCUS_HEADERS["weight"]:
            indices["weight"] = i
    return indices


def _read_ids_rows(ids_file: str, variant_mode: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if ids_file == "":
        return rows
    with _open_text(ids_file) as handle:
        try:
            first = next(handle)
        except StopIteration:
            return rows
        first = first.strip()
        if first == "":
            return rows
        delim = _detect_delim(first)
        parts = first.split(delim) if delim != " " else first.split()
        if variant_mode == "id":
            header_seen = parts[0] in ("varId", "VAR_ID", "VARID", "id")
            if not header_seen:
                upper = [p.strip().upper() for p in parts]
                if (
                    set(upper).intersection(_LOCUS_HEADERS["chrom"])
                    and set(upper).intersection(_LOCUS_HEADERS["pos"])
                    and set(upper).intersection(_LOCUS_HEADERS["ref"])
                    and set(upper).intersection(_LOCUS_HEADERS["alt"])
                ):
                    header_seen = True
            if not header_seen:
                weight = float(parts[1]) if len(parts) > 1 else 1.0
                rows.append((parts[0], weight))
            for line in handle:
                line = line.strip()
                if line == "":
                    continue
                parts = line.split(delim) if delim != " " else line.split()
                if header_seen and parts[0] in ("varId", "VAR_ID", "VARID", "id"):
                    continue
                weight = float(parts[1]) if len(parts) > 1 else 1.0
                rows.append((parts[0], weight))
            return rows

        if variant_mode == "auto":
            upper = [p.strip().upper() for p in parts]
            has_locus_header = (
                set(upper).intersection(_LOCUS_HEADERS["chrom"])
                and set(upper).intersection(_LOCUS_HEADERS["pos"])
                and set(upper).intersection(_LOCUS_HEADERS["ref"])
                and set(upper).intersection(_LOCUS_HEADERS["alt"])
            )
            if has_locus_header:
                variant_mode = "locus"
            else:
                seen_locus = False
                seen_id = False
                header_seen = parts[0] in ("varId", "VAR_ID", "VARID", "id")

                def handle_row(values: list[str]) -> None:
                    nonlocal seen_locus, seen_id
                    if not values:
                        return
                    parsed = _parse_locus_id(values[0])
                    weight = float(values[1]) if len(values) > 1 else 1.0
                    if parsed is None:
                        rows.append((values[0], weight))
                        seen_id = True
                    else:
                        chrom, pos, ref, alt = parsed
                        rows.append((_locus_key(chrom, pos, ref, alt), weight))
                        seen_locus = True

                if not header_seen:
                    handle_row(parts)
                for line in handle:
                    line = line.strip()
                    if line == "":
                        continue
                    values = line.split(delim) if delim != " " else line.split()
                    if header_seen and values and values[0] in ("varId", "VAR_ID", "VARID", "id"):
                        continue
                    handle_row(values)
                if seen_locus and seen_id:
                    print(
                        f"Warning: mixed variant ID formats detected in {ids_file}. "
                        "Parsed locus IDs when possible and kept raw IDs otherwise."
                    )
                return rows

        header = _parse_ids_header(parts)
        required = {"chrom", "pos", "ref", "alt"}
        if not required.issubset(header.keys()):
            raise new_error(
                "ids_file in locus mode must include columns for chrom, pos, ref, alt."
            )

        def parse_row(values: list[str]) -> tuple[str, float] | None:
            try:
                chrom = _normalize_chrom(values[header["chrom"]])
                pos = int(values[header["pos"]])
                ref = _normalize_allele(values[header["ref"]])
                alt = _normalize_allele(values[header["alt"]])
            except (IndexError, ValueError):
                return None
            weight = 1.0
            if "weight" in header:
                try:
                    weight = float(values[header["weight"]])
                except (IndexError, ValueError):
                    weight = 1.0
            return _locus_key(chrom, pos, ref, alt), weight

        for line in handle:
            line = line.strip()
            if line == "":
                continue
            values = line.split(delim) if delim != " " else line.split()
            parsed = parse_row(values)
            if parsed is None:
                continue
            rows.append(parsed)
    return rows


def read_id_keys(ids_file: str, variant_mode: str) -> list[str]:
    return [key for key, _weight in _read_ids_rows(ids_file, variant_mode)]


def load_ids(ids_file: str, n_traits: int, variant_mode: str) -> Dict[str, IdData]:
    beta_se_by_id: Dict[str, IdData] = {}
    if ids_file == "":
        return beta_se_by_id
    try:
        for id_value, weight in _read_ids_rows(ids_file, variant_mode):
            if id_value not in beta_se_by_id:
                beta_se_list = new_beta_se_list(n_traits)
                if weight < 0.0:
                    raise new_error(f"Negative weight ({weight}) for id {id_value}")
                beta_se_by_id[id_value] = IdData(beta_se_list, weight)
    except Exception as exc:
        raise for_file(ids_file, exc) from exc
    return beta_se_by_id


def _detect_locus_id_format(value: str) -> bool:
    return _VAR_ID_GUESS.match(value.strip()) is not None


def _parse_locus_id(value: str) -> tuple[str, int, str, str] | None:
    match = _VAR_ID_GUESS.match(value.strip())
    if not match:
        return None
    chrom = _normalize_chrom(match.group("chrom"))
    pos = int(match.group("pos"))
    ref = _normalize_allele(match.group("a1"))
    alt = _normalize_allele(match.group("a2"))
    return chrom, pos, ref, alt


def _detect_variant_mode_from_ids(ids_file: str) -> str:
    try:
        with _open_text(ids_file) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = re.split(r"[;\t, ]+", line)
                upper = [p.strip().upper() for p in parts]
                if set(upper).intersection(_LOCUS_HEADERS["chrom"]) and set(
                    upper
                ).intersection(_LOCUS_HEADERS["pos"]) and set(upper).intersection(
                    _LOCUS_HEADERS["ref"]
                ) and set(
                    upper
                ).intersection(
                    _LOCUS_HEADERS["alt"]
                ):
                    return "locus"
                if parts[0].upper() in ("VAR_ID", "VARID", "VARID", "ID", "VARID"):
                    continue
                return "locus" if _detect_locus_id_format(parts[0]) else "id"
    except Exception:
        return "id"
    return "id"


def _detect_variant_mode_from_gwas(config: "Config") -> str:
    if not config.gwas:
        return "id"
    gwas = config.gwas[0]
    file = _resolve_gwas_path(gwas, config.data_access)
    try:
        cache = _build_dig_cache(config.data_access)
        with _open_text(
            file,
            retries=config.data_access.retries,
            download=config.data_access.download,
            cache=cache,
        ) as handle:
            header = handle.readline().strip()
            first_line = handle.readline().strip()
    except Exception:
        return "id"
    if not header:
        return "id"
    delim = _detect_delim(header)
    parts = header.split(delim) if delim != " " else header.split()
    upper = {p.strip().upper() for p in parts}
    if (
        upper.intersection(_LOCUS_HEADERS["chrom"])
        and upper.intersection(_LOCUS_HEADERS["pos"])
        and upper.intersection(_LOCUS_HEADERS["ref"])
        and upper.intersection(_LOCUS_HEADERS["alt"])
    ):
        return "locus"
    if first_line:
        cols = gwas.cols or GwasCols(
            id=default_cols.VAR_ID, effect=default_cols.BETA, se=default_cols.SE
        )
        try:
            reader = GwasReader([header, first_line], cols, auto_meta=False)
            record = next(reader)
            if _detect_locus_id_format(record.var_id):
                return "locus"
        except Exception:
            pass
    return "id"


def resolve_variant_mode(config: "Config", action: Action) -> str:
    mode = config.variants.id_mode.lower()
    if mode in {"id", "locus"}:
        return mode
    if mode == "auto":
        return "auto"
    if action == Action.TRAIN:
        return _detect_variant_mode_from_ids(config.train.ids_file)
    return _detect_variant_mode_from_gwas(config)


def load_gwas(
    beta_se_by_id: Dict[str, IdData],
    gwas_config: "GwasConfig",
    n_traits: int,
    i_trait: int,
    action: Action,
    data_access: "DataAccessConfig",
    meta_by_id: dict[str, VariantMetaAccumulator] | None = None,
    auto_meta: bool = False,
    variant_mode: str = "id",
    flip_stats: dict[str, FlipStats] | None = None,
    bucket_id: int | None = None,
    n_buckets: int | None = None,
) -> None:
    file = _resolve_gwas_path(gwas_config, data_access)
    cols = gwas_config.cols or GwasCols(
        id=default_cols.VAR_ID, effect=default_cols.BETA, se=default_cols.SE
    )
    stats = None
    if flip_stats is not None:
        stats = flip_stats.get(gwas_config.name)
        if stats is None:
            stats = FlipStats()
            flip_stats[gwas_config.name] = stats
    mixed_seen_locus = False
    mixed_seen_id = False
    mixed_warned = False
    try:
        cache = _build_dig_cache(data_access)
        with _open_text(
            file,
            retries=data_access.retries,
            download=data_access.download,
            cache=cache,
        ) as handle:
            reader = GwasReader(handle, cols, auto_meta=auto_meta)
            first_record = None
            if variant_mode == "locus":
                missing_cols = []
                if reader._i_chr is None:
                    missing_cols.append("chrom")
                if reader._i_pos is None:
                    missing_cols.append("pos")
                if reader._i_effect_allele is None:
                    missing_cols.append("effect_allele")
                if reader._i_other_allele is None:
                    missing_cols.append("other_allele")
                if missing_cols:
                    try:
                        first_record = next(reader)
                    except StopIteration:
                        return
                    if not _detect_locus_id_format(first_record.var_id):
                        raise new_error(
                            "GWAS file must provide {} columns or locus-formatted IDs when variants.id_mode='locus'.".format(
                                ", ".join(missing_cols)
                            )
                        )

            def process_record(record: GwasRecord) -> None:
                nonlocal mixed_seen_locus, mixed_seen_id, mixed_warned
                beta = record.beta
                se = record.se
                flipped = False
                if variant_mode == "id":
                    var_id = record.var_id
                elif variant_mode == "auto":
                    parsed = None
                    if (
                        record.chrom is not None
                        and record.pos is not None
                        and record.effect_allele is not None
                        and record.other_allele is not None
                    ):
                        chrom = _normalize_chrom(record.chrom)
                        pos = record.pos
                        ref = _normalize_allele(record.other_allele)
                        alt = _normalize_allele(record.effect_allele)
                        parsed = (chrom, pos, ref, alt)
                    else:
                        parsed = _parse_locus_id(record.var_id)
                    if parsed is None:
                        mixed_seen_id = True
                        if mixed_seen_locus and not mixed_warned:
                            print(
                                f"Warning: mixed variant ID formats detected in GWAS {gwas_config.name}. "
                                "Parsed locus IDs when possible and kept raw IDs otherwise."
                            )
                            mixed_warned = True
                        var_id = record.var_id
                    else:
                        mixed_seen_locus = True
                        if mixed_seen_id and not mixed_warned:
                            print(
                                f"Warning: mixed variant ID formats detected in GWAS {gwas_config.name}. "
                                "Parsed locus IDs when possible and kept raw IDs otherwise."
                            )
                            mixed_warned = True
                        chrom, pos, ref, alt = parsed
                        key_direct = _locus_key(chrom, pos, ref, alt)
                        key_flip = _locus_key(chrom, pos, alt, ref)
                        if key_direct in beta_se_by_id:
                            var_id = key_direct
                        elif key_flip in beta_se_by_id:
                            var_id = key_flip
                            beta = -beta
                            flipped = True
                        elif action == Action.TRAIN and record.var_id in beta_se_by_id:
                            var_id = record.var_id
                            mixed_seen_id = True
                            if not mixed_warned:
                                print(
                                    f"Warning: mixed variant ID formats detected in GWAS {gwas_config.name}. "
                                    "Parsed locus IDs when possible and kept raw IDs otherwise."
                                )
                                mixed_warned = True
                        elif action == Action.CLASSIFY:
                            var_id = key_direct
                        else:
                            if stats is not None:
                                stats.missing_id += 1
                            return
                else:
                    parsed = None
                    if (
                        record.chrom is not None
                        and record.pos is not None
                        and record.effect_allele is not None
                        and record.other_allele is not None
                    ):
                        chrom = _normalize_chrom(record.chrom)
                        pos = record.pos
                        ref = _normalize_allele(record.other_allele)
                        alt = _normalize_allele(record.effect_allele)
                        parsed = (chrom, pos, ref, alt)
                    else:
                        parsed = _parse_locus_id(record.var_id)
                    if parsed is None:
                        if stats is not None:
                            stats.missing_meta += 1
                        return
                    chrom, pos, ref, alt = parsed
                    key_direct = _locus_key(chrom, pos, ref, alt)
                    key_flip = _locus_key(chrom, pos, alt, ref)
                    if key_direct in beta_se_by_id:
                        var_id = key_direct
                    elif key_flip in beta_se_by_id:
                        var_id = key_flip
                        beta = -beta
                        flipped = True
                    elif action == Action.CLASSIFY:
                        var_id = key_direct
                    else:
                        if stats is not None:
                            stats.missing_id += 1
                        return
                if bucket_id is not None and n_buckets is not None:
                    if _bucket_for_key(var_id, n_buckets) != bucket_id:
                        return
                if var_id in beta_se_by_id:
                    beta_se_by_id[var_id].beta_se_list[i_trait] = BetaSe(beta=beta, se=se)
                elif action == Action.CLASSIFY:
                    beta_se_list = new_beta_se_list(n_traits)
                    beta_se_list[i_trait] = BetaSe(beta=beta, se=se)
                    beta_se_by_id[var_id] = IdData(beta_se_list, 1.0)
                if stats is not None:
                    stats.matched += 1
                    if flipped:
                        stats.flipped += 1
                if meta_by_id is not None:
                    meta_record = record
                    if flipped:
                        meta_record = GwasRecord(
                            var_id=record.var_id,
                            beta=record.beta,
                            se=record.se,
                            chrom=record.chrom,
                            pos=record.pos,
                            effect_allele=record.other_allele,
                            other_allele=record.effect_allele,
                            eaf=record.eaf,
                            rsid=record.rsid,
                        )
                    accumulator = meta_by_id.get(var_id)
                    if accumulator is None:
                        accumulator = VariantMetaAccumulator()
                        meta_by_id[var_id] = accumulator
                    accumulator.add_record(meta_record, gwas_config.name, var_id)

            if first_record is not None:
                process_record(first_record)
            for record in reader:
                process_record(record)
    except Exception as exc:
        raise for_context(file, exc) from exc


def new_beta_se_list(n_traits: int) -> list[BetaSe]:
    return [BetaSe(beta=float("nan"), se=float("nan")) for _ in range(n_traits)]


def _build_dig_cache(data_access: "DataAccessConfig") -> object | None:
    if DigCacheConfig is None:
        return None
    if not data_access.cache_dir:
        return None
    max_bytes = data_access.cache_max_bytes
    ttl_days = data_access.cache_ttl_days
    return DigCacheConfig(
        dir=data_access.cache_dir,
        max_bytes=max_bytes,
        ttl_days=ttl_days,
    )


def _open_text(
    path: str,
    *,
    retries: int = 3,
    download: bool = False,
    cache: object | None = None,
) -> io.TextIOBase:
    if open_data_text is not None:
        return _IterableTextWrapper(
            open_data_text(
                path,
                encoding="utf-8",
                retries=retries,
                download=download,
                cache=cache,
            )
        )
    if path.startswith("file://"):
        path = path[len("file://") :]
    if "://" in path:
        raise new_error(
            "Remote URI provided but dig-open-data is not installed. "
            "Install it or use a local file path."
        )
    raw = open(path, "rb")
    magic = raw.read(2)
    raw.seek(0)
    if magic == b"\x1f\x8b":
        return _as_iterable_text(gzip.open(raw, "rt", encoding="utf-8"))
    return _as_iterable_text(io.TextIOWrapper(raw, encoding="utf-8"))


def _resolve_uri(path: str, base_uri: str | None) -> str:
    if base_uri is None:
        return path
    if "://" in path:
        return path
    base = base_uri.rstrip("/")
    rel = path.lstrip("/")
    return f"{base}/{rel}"


def _write_flip_log(path: str | None, flip_stats: dict[str, FlipStats]) -> None:
    if not path:
        return
    lines = ["gwas\tmatched\tflipped\tmissing_meta\tmissing_id"]
    for gwas_name in sorted(flip_stats.keys()):
        stats = flip_stats[gwas_name]
        lines.append(
            f"{gwas_name}\t{stats.matched}\t{stats.flipped}\t"
            f"{stats.missing_meta}\t{stats.missing_id}"
        )
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except Exception as exc:
        print(f"Warning: failed to write flip log {path}: {exc}")


def write_flip_log(path: str | None, flip_stats: dict[str, FlipStats]) -> None:
    _write_flip_log(path, flip_stats)


class _IterableTextWrapper:
    def __init__(self, handle: io.TextIOBase) -> None:
        self._handle = handle

    def __iter__(self):
        return self

    def __next__(self):
        line = self._handle.readline()
        if line == "":
            raise StopIteration
        return line

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._handle.close()

    def close(self) -> None:
        self._handle.close()

    def __getattr__(self, name):
        return getattr(self._handle, name)


def _as_iterable_text(handle: io.TextIOBase) -> io.TextIOBase:
    if hasattr(handle, "__iter__") and hasattr(handle, "__next__"):
        return handle
    return _IterableTextWrapper(handle)


_DIG_OPEN_DATA_URI_PREFIX = "dig-open-data:"


def _parse_dig_open_data_uri(uri: str) -> tuple[str, str] | None:
    if not uri.startswith(_DIG_OPEN_DATA_URI_PREFIX):
        return None
    parts = uri.split(":")
    if len(parts) != 3:
        raise new_error(
            f"Invalid dig-open-data URI {uri!r}. Expected format: "
            "'dig-open-data:<ancestry>:<trait>'."
        )
    ancestry = parts[1].strip()
    trait = parts[2].strip()
    if not ancestry or not trait:
        raise new_error(
            f"Invalid dig-open-data URI {uri!r}. Both ancestry and trait are required."
        )
    return ancestry, trait


def _resolve_gwas_path(gwas_config: "GwasConfig", data_access: "DataAccessConfig") -> str:
    uri = gwas_config.uri or gwas_config.file or ""
    parsed = _parse_dig_open_data_uri(uri)
    if parsed is not None:
        if dig_build_key is None or open_data_text is None:
            raise new_error(
                "dig-open-data URI provided but dig-open-data is not installed."
            )
        ancestry, trait = parsed
        key = dig_build_key(ancestry, trait)
        return f"s3://{DIG_DEFAULT_BUCKET}/{key}"
    provider = (data_access.provider or "").lower()
    if provider == "dig_open_data":
        if dig_build_key is None or open_data_text is None:
            raise new_error(
                "dig_open_data provider requested but dig-open-data is not installed."
            )
        if not gwas_config.trait:
            raise new_error(
                f"dig_open_data provider requires gwas.trait for {gwas_config.name}."
            )
        ancestry = data_access.ancestry
        if not ancestry:
            raise new_error("dig_open_data provider requires data_access.ancestry.")
        bucket = data_access.bucket or DIG_DEFAULT_BUCKET
        prefix = data_access.prefix or DIG_DEFAULT_PREFIX
        suffix = data_access.suffix or DIG_DEFAULT_SUFFIX
        key = dig_build_key(ancestry, gwas_config.trait, prefix=prefix, suffix=suffix)
        return f"s3://{bucket}/{key}"
    if not uri:
        raise new_error(
            f"GWAS entry {gwas_config.name} must set file or uri (or legacy trait/provider)."
        )
    return _resolve_uri(uri, data_access.gwas_base_uri)


_VAR_ID_GUESS = re.compile(
    r"^(?:chr)?(?P<chrom>[A-Za-z0-9]+)[^A-Za-z0-9]+(?P<pos>\d+)[^A-Za-z0-9]+(?P<a1>[A-Za-z]+)[^A-Za-z0-9]+(?P<a2>[A-Za-z]+)$",
    re.IGNORECASE,
)


def finalize_variant_meta(
    meta_by_id: dict[str, VariantMetaAccumulator],
    var_ids: list[str],
    allow_guess: bool,
    guess_order: str,
) -> dict[str, VariantMeta]:
    out: dict[str, VariantMeta] = {}
    n_guessed = 0
    for var_id in var_ids:
        accumulator = meta_by_id.get(var_id) or VariantMetaAccumulator()
        if allow_guess:
            guessed = _guess_variant_meta(var_id, guess_order)
            if guessed is not None:
                n_guessed += 1
                if accumulator.chrom is None:
                    accumulator.chrom = guessed.chrom
                if accumulator.pos is None:
                    accumulator.pos = guessed.pos
                if accumulator.effect_allele is None:
                    accumulator.effect_allele = guessed.effect_allele
                if accumulator.other_allele is None:
                    accumulator.other_allele = guessed.other_allele
        out[var_id] = accumulator.finalize()
    if allow_guess and n_guessed > 0:
        print(
            "Warning: GWAS-SSF fields guessed from variant IDs for "
            f"{n_guessed} variants (order={guess_order}). "
            "Set classify.gwas_ssf_guess_fields=false to disable."
        )
    return out


def _guess_variant_meta(var_id: str, guess_order: str) -> VariantMeta | None:
    match = _VAR_ID_GUESS.match(var_id)
    if not match:
        return None
    chrom = match.group("chrom")
    pos = int(match.group("pos"))
    a1 = match.group("a1")
    a2 = match.group("a2")
    if guess_order not in {"effect_other", "other_effect"}:
        raise new_error(
            f"classify.gwas_ssf_variant_id_order must be 'effect_other' or 'other_effect', got '{guess_order}'."
        )
    if guess_order == "other_effect":
        a1, a2 = a2, a1
    return VariantMeta(
        chrom=chrom,
        pos=pos,
        effect_allele=a1,
        other_allele=a2,
        rsid=None,
        eaf=None,
    )


def load_data_for_ids(config: "Config", ids: list[str]) -> GwasData:
    n_traits = len(config.gwas)
    beta_se_by_id: Dict[str, IdData] = {}
    for var_id in ids:
        beta_se_by_id[var_id] = IdData(new_beta_se_list(n_traits), 1.0)

    trait_names: list[str] = []
    variant_mode = resolve_variant_mode(config, Action.TRAIN)
    for i_trait, gwas in enumerate(config.gwas):
        trait_names.append(gwas.name)
        load_gwas(
            beta_se_by_id,
            gwas,
            n_traits,
            i_trait,
            Action.TRAIN,
            data_access=config.data_access,
            variant_mode=variant_mode,
        )

    n_data_points = len(beta_se_by_id)
    var_ids: list[str] = []
    betas = matrix_fill(n_data_points, n_traits, lambda _i, _j: np.nan)
    ses = matrix_fill(n_data_points, n_traits, lambda _i, _j: np.nan)

    for i_data_point, (var_id, id_data) in enumerate(beta_se_by_id.items()):
        var_ids.append(var_id)
        for i_trait, beta_se in enumerate(id_data.beta_se_list):
            betas[i_data_point, i_trait] = beta_se.beta
            ses[i_data_point, i_trait] = beta_se.se

    missing_by_trait: dict[str, list[str]] = {name: [] for name in trait_names}
    for i_data_point, var_id in enumerate(var_ids):
        for i_trait, trait_name in enumerate(trait_names):
            if np.isnan(betas[i_data_point, i_trait]) or np.isnan(ses[i_data_point, i_trait]):
                if len(missing_by_trait[trait_name]) < 5:
                    missing_by_trait[trait_name].append(var_id)
    missing_counts = {
        name: len(ids)
        for name, ids in missing_by_trait.items()
        if len(ids) > 0
    }
    if missing_counts:
        parts = []
        for name, count in missing_counts.items():
            examples = ", ".join(missing_by_trait[name])
            parts.append(f"{name}: {count} missing (e.g., {examples})")
        print(
            "Warning: missing GWAS entries for training IDs; "
            "treating missing traits as unobserved. "
            + " | ".join(parts)
        )

    endo_names = [item.name for item in config.endophenotypes]
    meta = Meta(trait_names=trait_names, var_ids=var_ids, endo_names=endo_names)
    return GwasData(meta=meta, betas=betas, ses=ses)


def subset_gwas_data(data: GwasData, indices: list[int]) -> GwasData:
    if not indices:
        raise new_error("Cannot subset GwasData with empty indices.")
    betas = data.betas[indices, :]
    ses = data.ses[indices, :]
    var_ids = [data.meta.var_ids[i] for i in indices]
    meta = Meta(
        trait_names=data.meta.trait_names,
        var_ids=var_ids,
        endo_names=data.meta.endo_names,
    )
    return GwasData(meta=meta, betas=betas, ses=ses)

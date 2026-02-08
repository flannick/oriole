from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import time
import gzip
import io

import numpy as np
import re
import math

from ..error import new_error, for_context, for_file
from ..math_utils.matrix import matrix_fill
from ..options.action import Action
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..options.config import Config, GwasConfig
from .gwas import GwasReader, GwasRecord, GwasCols, default_cols

try:
    from dig_open_data import open_text as open_data_text
except Exception:  # pragma: no cover - optional dependency
    open_data_text = None

DELIM_LIST = [";", "\t", ",", " "]


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
    if action == Action.TRAIN:
        print(f"Loading training IDs from {config.train.ids_file}")
        ids_start = time.perf_counter()
        beta_se_by_id = load_ids(config.train.ids_file, n_traits)
        print(
            f"Loaded {len(beta_se_by_id)} training IDs in "
            f"{time.perf_counter() - ids_start:.2f}s"
        )
    else:
        print("Loading full GWAS for classification (all IDs).")
        beta_se_by_id = {}

    trait_names: list[str] = []
    gwas_start = time.perf_counter()
    for i_trait, gwas in enumerate(config.gwas):
        trait_names.append(gwas.name)
        trait_start = time.perf_counter()
        load_gwas(
            beta_se_by_id,
            gwas,
            n_traits,
            i_trait,
            action,
            meta_by_id=meta_by_id,
            auto_meta=allow_meta_guess,
            base_uri=config.data_access.gwas_base_uri,
        )
        print(
            f"Loaded GWAS {gwas.name} from {gwas.file} in "
            f"{time.perf_counter() - trait_start:.2f}s"
        )
    print(f"Finished GWAS load in {time.perf_counter() - gwas_start:.2f}s")

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


def load_ids(ids_file: str, n_traits: int) -> Dict[str, IdData]:
    beta_se_by_id: Dict[str, IdData] = {}
    this_might_still_be_header = True
    splitter = re.compile(r"[;\t, ]+")
    try:
        with _open_text(ids_file) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = [part for part in splitter.split(line) if part != ""]
                if len(parts) == 1:
                    id_value = parts[0]
                    weight_str = None
                else:
                    id_value, weight_str = parts[0], parts[1]
                if id_value:
                    beta_se_list = new_beta_se_list(n_traits)
                    if weight_str is None:
                        weight = 1.0
                        this_might_still_be_header = False
                        beta_se_by_id[id_value] = IdData(beta_se_list, weight)
                    else:
                        try:
                            weight = float(weight_str)
                        except ValueError as exc:
                            if this_might_still_be_header:
                                continue
                            raise for_context(
                                f"Error parsing weight for id {id_value}", exc
                            ) from exc
                        this_might_still_be_header = False
                        if weight < 0.0:
                            raise new_error(f"Negative weight ({weight}) for id {id_value}")
                        beta_se_by_id[id_value] = IdData(beta_se_list, weight)
    except Exception as exc:
        raise for_file(ids_file, exc) from exc
    return beta_se_by_id


def load_gwas(
    beta_se_by_id: Dict[str, IdData],
    gwas_config: "GwasConfig",
    n_traits: int,
    i_trait: int,
    action: Action,
    meta_by_id: dict[str, VariantMetaAccumulator] | None = None,
    auto_meta: bool = False,
    base_uri: str | None = None,
) -> None:
    file = _resolve_uri(gwas_config.file, base_uri)
    cols = gwas_config.cols or GwasCols(
        id=default_cols.VAR_ID, effect=default_cols.BETA, se=default_cols.SE
    )
    try:
        with _open_text(file) as handle:
            reader = GwasReader(handle, cols, auto_meta=auto_meta)
            for record in reader:
                var_id = record.var_id
                beta = record.beta
                se = record.se
                if var_id in beta_se_by_id:
                    beta_se_by_id[var_id].beta_se_list[i_trait] = BetaSe(beta=beta, se=se)
                elif action == Action.CLASSIFY:
                    beta_se_list = new_beta_se_list(n_traits)
                    beta_se_list[i_trait] = BetaSe(beta=beta, se=se)
                    beta_se_by_id[var_id] = IdData(beta_se_list, 1.0)
                if meta_by_id is not None:
                    accumulator = meta_by_id.get(var_id)
                    if accumulator is None:
                        accumulator = VariantMetaAccumulator()
                        meta_by_id[var_id] = accumulator
                    accumulator.add_record(record, gwas_config.name, var_id)
    except Exception as exc:
        raise for_context(file, exc) from exc


def new_beta_se_list(n_traits: int) -> list[BetaSe]:
    return [BetaSe(beta=float("nan"), se=float("nan")) for _ in range(n_traits)]


def _open_text(path: str) -> io.TextIOBase:
    if open_data_text is not None:
        return open_data_text(path, encoding="utf-8")
    if "://" in path:
        raise new_error(
            "Remote URI provided but dig-open-data is not installed. "
            "Install it or use a local file path."
        )
    raw = open(path, "rb")
    magic = raw.read(2)
    raw.seek(0)
    if magic == b"\x1f\x8b":
        return gzip.open(raw, "rt", encoding="utf-8")
    return io.TextIOWrapper(raw, encoding="utf-8")


def _resolve_uri(path: str, base_uri: str | None) -> str:
    if base_uri is None:
        return path
    if "://" in path:
        return path
    base = base_uri.rstrip("/")
    rel = path.lstrip("/")
    return f"{base}/{rel}"


_VAR_ID_GUESS = re.compile(
    r"^(?:chr)?(?P<chrom>[A-Za-z0-9]+)[^A-Za-z0-9]+(?P<pos>\\d+)[^A-Za-z0-9]+(?P<a1>[A-Za-z]+)[^A-Za-z0-9]+(?P<a2>[A-Za-z]+)$",
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
    for i_trait, gwas in enumerate(config.gwas):
        trait_names.append(gwas.name)
        load_gwas(beta_se_by_id, gwas, n_traits, i_trait, Action.TRAIN)

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

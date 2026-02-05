from __future__ import annotations

import csv
from pathlib import Path
import math
import os

_mpl_dir = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/oriole_mplconfig"))
_mpl_dir.mkdir(parents=True, exist_ok=True)
_xdg_dir = Path(os.environ.setdefault("XDG_CACHE_HOME", "/tmp/oriole_xdg_cache"))
_xdg_dir.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_param_convergence(trace_file: str, out_file: str) -> None:
    path = Path(trace_file)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows = [row for row in reader if row]
    if len(header) < 2 or not rows:
        return
    indices = [int(row[0]) for row in rows]
    series = list(zip(*[[float(v) for v in row[1:]] for row in rows]))
    names = header[1:]

    n_params = len(names)
    ncols = 4
    nrows = int(math.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
    for i, name in enumerate(names):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        ax.plot(indices, series[i], linewidth=1)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("iteration", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)
    for j in range(n_params, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)


def plot_cv_grid(
    scores: dict[tuple[float, float], float],
    kappa_grid: list[float],
    pi_grid: list[float],
    chosen: tuple[float, float],
    out_file: str,
) -> None:
    if not scores:
        return
    kappa_sorted = list(kappa_grid)
    pi_sorted = list(pi_grid)
    z = [[scores.get((k, p), float("nan")) for k in kappa_sorted] for p in pi_sorted]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(z, aspect="auto", origin="lower")
    ax.set_xticks(range(len(kappa_sorted)))
    ax.set_xticklabels([str(k) for k in kappa_sorted], rotation=45, ha="right")
    ax.set_yticks(range(len(pi_sorted)))
    ax.set_yticklabels([str(p) for p in pi_sorted])
    ax.set_xlabel("kappa")
    ax.set_ylabel("pi")
    ax.set_title("Cross-validation score grid")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if chosen in scores:
        j = kappa_sorted.index(chosen[0])
        i = pi_sorted.index(chosen[1])
        ax.scatter([j], [i], s=80, facecolors="none", edgecolors="red", linewidths=2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)

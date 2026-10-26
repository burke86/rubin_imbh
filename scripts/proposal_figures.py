"""Figures for the NSF proposal methods section (no figure titles; PDFs).

1. pipeline.pdf: simulate -> observe like DP2 -> train the flow (weighted by empirical priors)
   -> posteriors -> validation, with the loop back to the model and priors.
2. compute_time.pdf: grouped bars of the CPU core-hours to process DP2 and DR1 with per-object
   MAP + Laplace fitting, batched (vectorized) fitting and NPE, from per-object speeds measured in this project, with
   the wall-clock time on a 1,000-core cluster beside each.

Catalogue sizes: DP2 ~7.9e8 objects (RTN-115 draft), DR1 2.3e10 (DMTN-048 planning estimate).

usage: python scripts/proposal_figures.py --out data/sims/plots/proposal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from jaxsedfit.mplstyle import use_style  # noqa: E402

CATALOGUES = {"DP2": 7.9e8, "DR1": 2.3e10}
# (label, core-seconds per object, one-time core-hours). NPE: measured, 5.6 ms wall per object
# (100 draws) at 3.25 cores in use; one-time = 1.5 h to simulate and train at the same load.
# Fitting: 16 s wall per single-object MAP + Laplace fit, counted as one core (favouring it);
# batched fitting 0.375 s wall per object, counted at the same 3.25 cores as NPE.
METHODS = [
    ("Per-object fitting (MAP + Laplace)", 16.0, 0.0),
    ("Batched fitting (vectorized MAP + Laplace)", 0.375 * 3.25, 0.0),
    ("NPE, including simulation and training", 0.0056 * 3.25, 1.5 * 3.25),
]
CLUSTER_CORES = 1000
HOURS_PER_YEAR = 8766.0


def pipeline(out: Path) -> Path:
    boxes = {  # name: (x centre, y centre, title, subtitle)
        "model": (1.0, 2.0, "Forward model", "jaxsedfit host + AGN"),
        "observe": (3.0, 2.0, "Observe like DP2", "DP2 noise and cadence"),
        "train": (5.0, 2.0, "Train the flow", "1.6M weighted sims"),
        "post": (7.0, 2.0, "Posteriors", "z, M$_\\star$, L$_{\\rm AGN}$ draws"),
        "priors": (5.0, 0.8, "Empirical priors", "COSMOS2020, SARDF"),
        "valid": (7.0, 0.8, "Validation", "spec-z, X-ray, quasars"),
    }
    w, h = 1.7, 0.7
    with use_style():
        fig, ax = plt.subplots(figsize=(9.0, 3.0))
        for key, (x, y, title, sub) in boxes.items():
            main = key == "train"
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor="C0" if main else "white", alpha=0.15 if main else 1.0,
                                        edgecolor="none"))
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor="none", edgecolor="C0" if main else "0.45", lw=2 if main else 1.1))
            ax.text(x, y + 0.11, title, ha="center", va="center", fontsize=11, fontweight="bold")
            ax.text(x, y - 0.16, sub, ha="center", va="center", fontsize=8.5, color="0.35")

        def arrow(p0, p1, **kw):
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, color="0.45", lw=1.1,
                                         shrinkA=0, shrinkB=0, **kw))

        for a, b in (("model", "observe"), ("observe", "train"), ("train", "post")):
            arrow((boxes[a][0] + w / 2, 2.0), (boxes[b][0] - w / 2, 2.0))
        arrow((5.0, 0.8 + h / 2), (5.0, 2.0 - h / 2))
        ax.text(5.08, 1.4, "weights", fontsize=8.5, color="0.35", va="center")
        arrow((7.0, 2.0 - h / 2), (7.0, 0.8 + h / 2))
        # dashed loop: validation -> forward model
        ax.plot([7.0, 7.0, 1.0], [0.8 - h / 2, 0.1, 0.1], color="0.45", lw=1.1, ls="--")
        arrow((1.0, 0.1), (1.0, 2.0 - h / 2), linestyle="--")
        ax.text(3.0, 0.18, "revise the model or priors only if validation improves", ha="center", va="bottom",
                fontsize=8.5, color="0.35")
        ax.set_xlim(0.0, 8.0)
        ax.set_ylim(-0.05, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout()
        fn = out / "pipeline.pdf"
        fig.savefig(fn, bbox_inches="tight")
        plt.close(fig)
    return fn


def fmt_time(hours: float) -> str:
    """Wall-clock words: hours, days or years."""
    if hours < 48:
        return f"{hours:.2g} h" if hours < 10 else f"{hours:.0f} h"
    if hours < 1461:
        return f"{hours / 24:.0f} days"
    years = hours / HOURS_PER_YEAR
    return f"{years:.2g} yr" if years < 100 else f"{float(f'{years:.2g}'):,.0f} yr"


def sci(x: float) -> str:
    e = int(np.floor(np.log10(x)))
    m = x / 10**e
    return rf"{m:.1f}$\times$10$^{{{e}}}$"


def compute_time(out: Path) -> Path:
    """Core-hours to process DP2 and DR1 per method, as grouped bars on a log axis; each bar is
    labelled with its core-hours and the wall-clock time on a 1,000-core cluster."""
    colors = ["0.72", "0.5", "C0"]
    width = 0.26
    with use_style():
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        cats = list(CATALOGUES)
        for j, (label, core_sec, fixed) in enumerate(METHODS):
            npe = label.startswith("NPE")
            xs = np.arange(len(cats)) + (j - 1) * width
            vals = np.array([CATALOGUES[c] * core_sec / 3600 + fixed for c in cats])
            ax.bar(xs, vals, width=width * 0.92, bottom=0, color=colors[j], label=label, zorder=2)
            for x, v in zip(xs, vals):
                ax.text(x, v * 1.6, f"{sci(v)}\n{fmt_time(v / CLUSTER_CORES)}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold" if npe else "normal", linespacing=1.15)
        ax.set_yscale("log")
        ax.set_ylim(1e2, 3e10)
        ax.set_yticks([1e2, 1e4, 1e6, 1e8, 1e10])
        ax.set_yticklabels([r"10$^2$", r"10$^4$", r"10$^6$", r"10$^8$", r"10$^{10}$"])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels([rf"{c}" + "\n" + rf"{sci(CATALOGUES[c])} objects" for c in cats], fontsize=10)
        ax.tick_params(which="both", top=False, right=False, bottom=False)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)
        for side in ("right", "top"):
            ax.spines[side].set_visible(False)
        ax.set_ylabel("CPU core-hours", fontsize=11)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)
        ax.text(1.0, 1.01, f"labels: core-hours, wall-clock time on {CLUSTER_CORES:,} cores", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color="0.35")
        fig.tight_layout()
        fn = out / "compute_time.pdf"
        fig.savefig(fn, bbox_inches="tight")
        plt.close(fig)
    return fn


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/sims/plots/proposal")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (pipeline(out), compute_time(out)):
        print(fn)


if __name__ == "__main__":
    main()

"""Figures for the NSF proposal methods section (no figure titles; PDFs).

1. pipeline.pdf: simulate -> observe like DP2 -> train the flow (weighted by empirical priors)
   -> posteriors -> validation, with the loop back to the model and priors.
2. compute_time.pdf: CPU core-hours to process DP2 and DR1 with per-object MAP + Laplace fitting,
   batched (vectorized) fitting and NPE, from per-object speeds measured in this project, with
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
    """Core-hours to process DP2 and DR1 per method; labels give the wall-clock time on a 1,000-core cluster."""
    with use_style():
        fig, ax = plt.subplots(figsize=(8.0, 3.9))
        y, yticks, ylabels = 0.0, [], []
        for cat, n in CATALOGUES.items():
            exp = int(np.floor(np.log10(n)))
            ax.annotate(rf"{cat}: {n / 10**exp:.1f} $\times$ 10$^{{{exp}}}$ objects", xy=(0.0, y),
                        xycoords=("axes fraction", "data"), xytext=(-5, 0), textcoords="offset points",
                        ha="right", va="center", fontweight="bold", fontsize=10.5)
            y -= 0.6
            for label, core_sec, fixed in METHODS:
                core_h = n * core_sec / 3600 + fixed
                npe = label.startswith("NPE")
                color = "C0" if npe else "0.6"
                ax.plot([1e2, core_h], [y, y], color=color, lw=1.3, alpha=1.0 if npe else 0.7)
                ax.plot([core_h], [y], "o", color=color, ms=7)
                ax.text(core_h * 1.7, y, rf"{sci(core_h)} core-h  $\cdot$  {fmt_time(core_h / CLUSTER_CORES)}",
                        va="center", fontsize=9.5, fontweight="bold" if npe else "normal")
                yticks.append(y)
                ylabels.append(label)
                y -= 0.6
            y -= 0.35
        ax.set_xscale("log")
        ax.set_xlim(1e2, 3e11)
        ax.set_xticks([1e3, 1e5, 1e7, 1e9])
        ax.set_xticklabels([r"10$^3$", r"10$^5$", r"10$^7$", r"10$^9$"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.tick_params(which="both", top=False, right=False, left=False)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=9.5)
        for t, lab in zip(ax.get_yticklabels(), ylabels):
            if lab.startswith("NPE"):
                t.set_fontweight("bold")
        ax.set_ylim(y + 0.2, 0.4)
        ax.grid(axis="x", color="0.9", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("CPU core-hours", fontsize=10.5)
        ax.text(1.0, 1.02, f"labels: core-hours $\\cdot$ wall-clock time on {CLUSTER_CORES:,} cores", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="0.35")
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

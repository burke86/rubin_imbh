"""Figures for the NSF proposal methods section (no figure titles; PDFs).

1. pipeline.pdf: simulate -> observe like DP2 -> train the flow (weighted by empirical priors)
   -> posteriors -> validation, with the loop back to the model and priors.
2. compute_time.pdf: grouped bars of the CPU core-hours to process DP2 and DR1 with per-object
   MAP + Laplace fitting, batched (vectorized) fitting and NPE, from per-object speeds measured in this project, with
   the wall-clock time on a 1,000-core cluster beside each.
3. one_shot.pdf: cartoon contrasting the traditional sequence (classify AGN, then photo-z, then SED
   fit, each step passing on a point estimate) with the single joint posterior of NPE.

Catalog sizes: DP2 ~7.9e8 objects (RTN-115 draft), DR1 2.3e10 (DMTN-048 planning estimate).

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
# Fitting: 16 s wall per single-object MAP + Laplace fit, counted as one core (favoring it);
# batched fitting 0.375 s wall per object, counted at the same 3.25 cores as NPE.
METHODS = [
    ("Per-object fitting (MAP + Laplace)", 16.0, 0.0),
    ("Batched fitting (vectorized MAP + Laplace)", 0.375 * 3.25, 0.0),
    (r"NPE, simulation+training ($\mathbf{this\ proposal}$)", 0.0056 * 3.25, 1.5 * 3.25),
]
CLUSTER_CORES = 1000
HOURS_PER_YEAR = 8766.0


def pipeline(out: Path) -> Path:
    boxes = {  # name: (x center, y center, title, subtitle)
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


def _gauss2(x, y, mx, my, sx, sy, rho):
    u, v = (x - mx) / sx, (y - my) / sy
    return np.exp(-(u**2 - 2 * rho * u * v + v**2) / (2 * (1 - rho**2)))


# cartoon joint posterior of a faint variable AGN: a low-z, host-dominated solution and a high-z,
# AGN-dominated one; (weight, means (z, log M*, f_AGN), sigmas, correlations zm, zf, mf)
MODES = [(0.6, (0.7, 9.6, 0.35), (0.12, 0.35, 0.12), (0.6, 0.5, 0.3)),
         (0.4, (2.1, 10.4, 0.8), (0.18, 0.45, 0.07), (0.5, 0.4, 0.2))]
TRUTH = (2.0, 10.3, 0.78)
LIMS = [(0.0, 2.8), (8.4, 11.6), (0.0, 1.05)]
NAMES = [r"$z$", r"$\log M_\star$", r"$f_{\rm AGN}$"]


def _thin_frame(a, color="0.55"):
    a.set_xticks([])
    a.set_yticks([])
    a.tick_params(which="both", left=False, right=False, top=False, bottom=False)
    for sp in a.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color(color)


def _posterior_row(fig, rect, color):
    """p(z), then the z-log M* and z-f_AGN joint posteriors, side by side in figure coordinates ``rect``."""
    x0, y0, w, h = rect
    n, gap = 3, 0.03
    cw = (w - gap * (n - 1)) / n
    g = np.linspace(*LIMS[0], 200)
    for k, other in enumerate((None, 1, 2)):
        ax = fig.add_axes((x0 + k * (cw + gap), y0, cw, h))
        _thin_frame(ax)
        ax.set_xlim(*LIMS[0])
        ax.set_xlabel(NAMES[0], fontsize=8.5, labelpad=1)
        if other is None:
            p = sum(wt * np.exp(-0.5 * ((g - mu[0]) / sg[0]) ** 2) / sg[0] for wt, mu, sg, _ in MODES)
            ax.fill_between(g, p, color=color, alpha=0.25, lw=0)
            ax.plot(g, p, color=color, lw=1.1)
            ax.axvline(TRUTH[0], color="k", lw=0.8, ls=":")
            ax.set_ylim(0, p.max() * 1.15)
            ax.set_ylabel(r"$p(z)$", fontsize=8.5, labelpad=1)
            continue
        X, Y = np.meshgrid(g, np.linspace(*LIMS[other], 200))
        p = sum(wt * _gauss2(X, Y, mu[0], mu[other], sg[0], sg[other], rho[other - 1]) / (sg[0] * sg[other])
                for wt, mu, sg, rho in MODES)
        lev = p.max() * np.array([0.08, 0.35, 0.7, 1.01])
        ax.contourf(X, Y, p, levels=lev, colors=[color], alpha=0.18)
        ax.contour(X, Y, p, levels=lev[:-1], colors=[color], linewidths=0.8)
        ax.plot(TRUTH[0], TRUTH[other], marker="*", color="k", ms=5.5, mew=0)
        ax.set_ylim(*LIMS[other])
        ax.set_ylabel(NAMES[other], fontsize=8.5, labelpad=1)


def one_shot(out: Path) -> Path:
    """Traditional sequential inference vs the single joint posterior of NPE (cartoon)."""
    grey, warn, blue = "0.45", "#b2182b", "C0"
    with use_style():
        fig = plt.figure(figsize=(7.2, 2.95))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 10)
        ax.set_ylim(0.35, 4.45)
        ax.axis("off")

        def box(x, y, w, h, edge, face="white", lw=1.1):
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor=face, edgecolor=edge, lw=lw))

        def arrow(x0, x1, y, color=grey, label=None):
            ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=10, color=color, lw=1.1,
                                         shrinkA=0, shrinkB=0))
            if label:
                ax.text((x0 + x1) / 2, y + 0.07, label, ha="center", va="bottom", fontsize=7.5, color="0.25")

        def inset(x, y, w, h):
            f = fig.transFigure.inverted()
            (fx, fy), (fx1, fy1) = f.transform(ax.transData.transform([(x, y), (x + w, y + h)]))
            a = fig.add_axes((fx, fy, fx1 - fx, fy1 - fy))
            _thin_frame(a)
            return a

        rng = np.random.default_rng(3)
        lam = np.array([0.37, 0.48, 0.62, 0.75, 0.87, 0.97])  # ugrizy (micron)

        def sed_icon(x, y, w, h):
            a = inset(x, y, w, h)
            f = 0.9 * (lam / 0.6) ** -1.3 + 1.1 * (lam / 0.6) ** 1.5  # AGN power law + host
            a.errorbar(lam, f + rng.normal(0, 0.04, lam.size), yerr=0.12, fmt="o", ms=2.5, color="k",
                       elinewidth=0.7, capsize=0)
            a.set_xlim(0.3, 1.04)
            a.set_ylim(1.2, 4.2)

        # ---- traditional: three sequential steps
        yt, bw, bh = 3.45, 1.75, 0.9
        ax.text(0.2, 4.22, "Traditional: three sequential steps", fontsize=10.5, fontweight="bold", va="center")
        sed_icon(0.25, yt - 0.35, 0.85, 0.7)
        ax.text(0.68, yt - 0.42, "photometry", ha="center", va="top", fontsize=7.5, color="0.3")
        steps = [(2.3, "1. Classify", "AGN or galaxy?\n(color cut)"),
                 (4.85, "2. Photo-z", "templates set\nby the class"),
                 (7.4, "3. SED fit", r"$M_\star$, SFR at" + "\n" + r"fixed $z_{\rm best}$")]
        for x, title, sub in steps:
            box(x, yt, bw, bh, grey)
            ax.text(x, yt + 0.18, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
            ax.text(x, yt - 0.17, sub, ha="center", va="center", fontsize=7.5, color="0.3", linespacing=1.1)
        arrow(1.18, 2.3 - bw / 2 - 0.03, yt)
        arrow(2.3 + bw / 2 + 0.03, 4.85 - bw / 2 - 0.03, yt, label="label")
        arrow(4.85 + bw / 2 + 0.03, 7.4 - bw / 2 - 0.03, yt, label=r"$z_{\rm best}$")
        arrow(7.4 + bw / 2 + 0.03, 8.75, yt)
        for x, txt in ((3.575, "misclassification\nnot carried forward"),
                       (6.125, "photo-z errors and\ndegeneracies dropped")):
            ax.text(x, yt - 0.1, "✕", ha="center", va="center", fontsize=9, color=warn)
            ax.text(x, yt - bh / 2 - 0.08, txt, ha="center", va="top", fontsize=7.5, color=warn, linespacing=1.1)
        a = inset(8.85, yt - 0.35, 0.95, 0.7)  # an overconfident single mass estimate that misses the truth
        g = np.linspace(8.4, 11.6, 300)
        pm = np.exp(-0.5 * ((g - 9.55) / 0.12) ** 2)
        a.fill_between(g, pm, color=grey, alpha=0.25, lw=0)
        a.plot(g, pm, color=grey, lw=1.1)
        a.axvline(TRUTH[1], color="k", ls=":", lw=0.8)
        a.set_xlim(8.4, 11.6)
        a.set_ylim(0, 1.2)
        ax.text(9.32, yt - 0.42, r"$M_\star \pm \sigma$", ha="center", va="top", fontsize=7.5, color="0.3")
        ax.text(9.32, yt - bh / 2 - 0.22, "too narrow,\nbiased", ha="center", va="top", fontsize=7.5, color=warn,
                linespacing=1.1)

        # ---- this proposal: one shot (same height as the row above)
        ax.plot([0.2, 9.8], [2.33, 2.33], color="0.82", lw=0.8)
        ax.text(0.2, 2.1, "This proposal: one shot", fontsize=10.5, fontweight="bold", va="center", color=blue)
        yb = 1.12
        sed_icon(0.2, yb - 0.3, 0.68, 0.62)
        a = inset(0.95, yb - 0.3, 0.68, 0.62)
        t = np.sort(rng.uniform(0, 1, 25))
        lc = np.cumsum(rng.normal(0, 0.2, t.size))
        a.errorbar(t, lc - lc.mean(), yerr=0.12, fmt="o", ms=1.8, color="k", elinewidth=0.6, capsize=0)
        ax.text(0.915, yb - 0.37, "photometry + variability", ha="center", va="top", fontsize=7.5, color="0.3")
        bx, bw2 = 3.27, 2.85
        box(bx, yb, bw2, 1.05, blue, face="#eef4fb", lw=1.6)
        ax.text(bx, yb + 0.27, "Neural posterior estimator", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(bx, yb - 0.15, "one population of host + AGN\nsimulations, no AGN / galaxy\nsplit; trained once",
                ha="center", va="center", fontsize=7.5, color="0.3", linespacing=1.1)
        arrow(1.68, bx - bw2 / 2 - 0.03, yb, color=blue)
        arrow(bx + bw2 / 2 + 0.03, 4.95, yb, color=blue)
        f = fig.transFigure.inverted()
        (fx, fy), (fx1, fy1) = f.transform(ax.transData.transform([(5.25, 0.6), (7.6, 1.62)]))
        _posterior_row(fig, (fx, fy, fx1 - fx, fy1 - fy), blue)
        ax.text(7.75, 1.72, "single joint posterior", fontsize=9.5, fontweight="bold", color=blue, va="top")
        ax.text(7.75, 1.45, r"$p(z, M_\star, f_{\rm AGN}, \ldots \mid {\rm data})$", fontsize=8.5, color=blue,
                va="top")
        ax.text(7.75, 1.12, "• AGN class → posterior of $f_{\\rm AGN}$\n• photo-z degeneracies kept\n"
                "• errors fully propagated", fontsize=7.5, va="top", color="0.2", linespacing=1.2)
        ax.plot([7.83], [0.42], marker="*", color="k", ms=5.5, mew=0)
        ax.text(7.95, 0.42, "truth", fontsize=7.5, va="center", color="0.2")
        fn = out / "one_shot.pdf"
        fig.savefig(fn)
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
    labeled with its core-hours and the wall-clock time on a 1,000-core cluster."""
    colors = ["0.72", "0.5", "C0"]
    width = 0.26
    with use_style():
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        cats = list(CATALOGUES)
        for j, (label, core_sec, fixed) in enumerate(METHODS):
            npe = label.startswith("NPE")
            xs = np.arange(len(cats)) + (j - 1) * width
            vals = np.array([CATALOGUES[c] * core_sec / 3600 + fixed for c in cats])
            ax.bar(xs, vals, width=width * 0.92, bottom=0, color=colors[j], label=label, zorder=2)
            for x, v in zip(xs, vals):
                ax.text(x, v * 1.6, f"{sci(v)}\nCPU hours\n({fmt_time(v / CLUSTER_CORES)})", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold" if npe else "normal", linespacing=1.15)
        ax.set_yscale("log")
        ax.set_ylim(1e3, 3e12)
        ax.set_yticks([1e4, 1e6, 1e8, 1e10, 1e12])
        ax.set_yticklabels([r"10$^4$", r"10$^6$", r"10$^8$", r"10$^{10}$", r"10$^{12}$"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels([rf"LSST {c}" + "\n" + rf"{sci(CATALOGUES[c])} objects" for c in cats], fontsize=10)
        ax.tick_params(axis="x", which="both", top=False, bottom=False)  # categories; y ticks on both sides
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.set_ylabel("CPU hours", fontsize=11)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)
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
    for fn in (pipeline(out), compute_time(out), one_shot(out)):
        print(fn)


if __name__ == "__main__":
    main()

"""Choose per-band ``--sys-frac`` floors from a ``--no-select --stats-only`` cone run.

    python scripts/fit_floor.py ~/ecdfs_stats_v3.parquet

The run must have used ``--sys-frac 0``. For each trial floor s, every object's chi^2 is
rescaled as if each epoch error had s * |flux| added in quadrature,

    chi2' ~= chi2 * rmsErr^2 / (rmsErr^2 + (s * medFlux)^2),

an approximation (it uses the median flux and rms error rather than per-epoch values),
so confirm the chosen floors with a real rerun. Reported per magnitude bin: the
5-sigma fraction of the 1-2 detection reference group (false positives, plus any real
faint variables) and of the >5 group (the cost in sensitivity).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.calib import calibration_groups, rescaled_sig  # noqa: E402

SIG = 5.0
BANDS = "griz"
FLOORS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
MAG_BINS = [-np.inf, 18, 19, 20, 21, 22, 23, 25, np.inf]
BRIGHT = 21.0  # floor is chosen to flatten the false-positive rate brighter than this


d = pd.read_parquet(sys.argv[1])
d["grp"], REF, VAR = calibration_groups(d)
print("reference group:", REF or "all objects (object source: nearly all constant)",
      "| likely-variable group:", VAR or "none")
pd.set_option("display.width", 200)
choice = {}
for b in BANDS:
    x = d[d[f"{b}_nEpoch"] >= 10].copy()
    if len(x) == 0:
        print(f"\n=== {b}: no objects with >= 10 epochs, skipped")
        continue
    with np.errstate(invalid="ignore", divide="ignore"):
        x["mag"] = 31.4 - 2.5 * np.log10(x[f"{b}_medFlux"])
    x["magbin"] = pd.cut(x["mag"], MAG_BINS).astype(str)
    ref = (x.grp == REF) if REF else pd.Series(True, index=x.index)
    var = (x.grp == VAR) if VAR else pd.Series(False, index=x.index)
    # false-positive level the bright end should come down to: the faint reference bins,
    # where the floor is negligible next to the measurement errors
    faint = ref & x.mag.between(BRIGHT, 25)
    rows_ref, rows_var = {}, {}
    for s in FLOORS:
        hit = pd.Series(rescaled_sig(x, b, s) >= SIG, index=x.index)
        rows_ref[s] = hit[ref].groupby(x.magbin[ref]).mean()
        rows_ref[s]["ALL"] = hit[ref].mean()
        rows_var[s] = hit[var].groupby(x.magbin[var]).mean()
        rows_var[s]["ALL"] = hit[var].mean()
    t_ref = pd.DataFrame(rows_ref).T
    t_var = pd.DataFrame(rows_var).T
    order = [c for c in map(str, pd.cut([], MAG_BINS).categories) if c in t_ref.columns] + ["ALL"]
    print(f"\n=== {b}: 5-sigma fraction, reference group ({REF or 'all objects'}; rows = floor)")
    print(t_ref[order].round(3).to_string())
    if VAR:
        print(f"--- {b}: 5-sigma fraction, likely-variable group ({VAR})")
        print(t_var[[c for c in order if c in t_var.columns]].round(3).to_string())

    baseline = (rescaled_sig(x[faint], b, 0.0) >= SIG).mean() if faint.any() else 0.0
    bright_cols = [c for c in order[:-1] if pd.Interval(*map(float, c.strip("(]").split(", "))).right <= BRIGHT]
    ok = [s for s in FLOORS if (t_ref.loc[s, bright_cols] <= max(baseline, 0.02)).all()]
    choice[b] = ok[0] if ok else None
    print(f"faint (21-25 mag) reference baseline: {baseline:.3f}; "
          f"smallest floor bringing every bright bin to <= max(baseline, 0.02): {choice[b]}")

good = {b: s for b, s in choice.items() if s is not None}
if good:
    print("\nsuggested: --sys-frac " + ",".join(f"{b}={s}" for b, s in good.items()))

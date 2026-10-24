"""Reference-catalogue matching for real-data validation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubin_imbh.references import match_reference  # noqa: E402


def test_match_reference_nearest_within_radius():
    rng = np.random.default_rng(9)
    # spread over 2 x 2 deg so unmatched objects have a negligible chance of a 1" coincidence
    obs = pd.DataFrame({"objectId": np.arange(300), "coord_ra": rng.uniform(149, 151, 300), "coord_dec": rng.uniform(1, 3, 300)})
    ref = pd.DataFrame({"ra": obs.coord_ra[:100] + 0.2 / 3600, "dec": obs.coord_dec[:100], "z_ref": np.arange(100) / 10.0,
                        "z_type": "spec"})
    m = match_reference(obs, ref, 1.0)
    assert np.isfinite(m.ref_z_ref[:100]).all() and np.allclose(m.ref_z_ref[:100], np.arange(100) / 10.0)
    assert m.ref_z_ref[100:].isna().all()

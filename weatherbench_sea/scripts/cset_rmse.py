"""Traditional RMSE for AIFS (12 cases) computed with CSET operators, compared with weatherbench.

Uses only CSET's read_cubes, constraint operators and scores_rmse on the CF files from prep_cf.py.
weatherbench averages per-case RMSEs, so RMSE is computed per valid time (time preserved), then
averaged over the 12 cases.

Usage: python cset_rmse.py PREPPED_DIR WEATHERBENCH_NPZ
"""

import sys
import warnings

import numpy as np

from CSET.operators import constraints, read, scoreswrappers

warnings.filterwarnings("ignore")

PREPPED, NPZ = sys.argv[1], sys.argv[2]
FIELDS = {
    "z": ("geopotential", 500),
    "t": ("air_temperature", 850),
    "q": ("specific_humidity", 700),
    "u": ("eastward_wind", 850),
    "v": ("northward_wind", 850),
}
ref = np.load(NPZ, allow_pickle=True)["traditional"].item()


def per_case_rmse(lead, varname, level):
    """RMSE of AIFS against ERA5 for each case, via read_cubes and scores_rmse."""
    constraint = constraints.combine_constraints(
        variable_constraint=constraints.generate_var_constraint(varname),
        level_constraint=constraints.generate_level_constraint("pressure", level),
    )
    cubes = read.read_cubes(
        [f"{PREPPED}/era5", f"{PREPPED}/aifs_lead{lead:02d}"],
        constraint=constraint,
        model_names=["ERA5", "AIFS"],
    )
    score = scoreswrappers.scores_rmse(cubes, preserved_coordinates=["time"])
    return np.asarray(score.data, dtype=float)


print(
    f"{'var':6}{'lead':>5}{'CSET':>14}{'weatherbench':>14}{'rel diff':>11}{'cases':>7}"
)
worst = 0.0
for lead in [12, 24, 36, 48]:
    per = {k: per_case_rmse(lead, n, lev) for k, (n, lev) in FIELDS.items()}
    out = {k: per[k].mean() for k in ("z", "t", "q")}
    out["wind"] = np.mean(np.sqrt(per["u"] ** 2 + per["v"] ** 2))
    for k, lev in [("z", 500), ("t", 850), ("q", 700), ("wind", 850)]:
        r = ref[k][lev][lead]
        rel = (out[k] - r) / r
        worst = max(worst, abs(rel))
        n = per["u"].size if k == "wind" else per[k].size
        print(f"{k:6}{lead:>5}{out[k]:>14.6g}{r:>14.6g}{rel:>11.2e}{n:>7}")
print(
    f"\nworst |relative difference|: {worst:.2e}  ->  {'PASS' if worst < 1e-4 else 'FAIL'} (threshold 1e-4)"
)

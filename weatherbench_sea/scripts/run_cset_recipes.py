"""Bake the WeatherBench Dynamic and FSS recipes for every lead and threshold.

Does what `cset bake -i ... -r ... --VAR=value` does (parse_variable_options,
parse_recipe, execute_recipe) for each job, in one process to avoid paying the
start-up cost 48 times. Writes every per-case value to a JSON file.

Usage: WB_CSET_WORK=<work dir> HDF5_USE_FILE_LOCKING=FALSE python run_cset_recipes.py OUT_JSON
"""

import glob
import json
import os
import sys
import warnings
from pathlib import Path

import iris

from CSET._common import parse_recipe, parse_variable_options
from CSET.operators import execute_recipe

warnings.filterwarnings("ignore")
# Data and outputs live in a work directory outside the repo: WB_CSET_WORK, default the current
# directory. It holds prepped/ (from prep_cf.py), reference/ and the bake outputs.
WORK = Path(os.environ.get("WB_CSET_WORK", Path.cwd()))
RECIPES = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "CSET"
    / "recipes"
    / "weatherbench_sea"
)
PREP, BAKE = WORK / "prepped", WORK / "bake_out_dyn_fss"

DYNAMIC = {  # weatherbench metric: (recipe, recipe variables, truth dir, model dir prefix)
    "mse_convergence": (
        "wb_dyn_mse_flux.yaml",
        ["--LEVEL=850", "--SIGN=negative"],
        "era5",
        "aifs_lead",
    ),
    "mse_divergence": (
        "wb_dyn_mse_flux.yaml",
        ["--LEVEL=200", "--SIGN=positive"],
        "era5",
        "aifs_lead",
    ),
    "vorticity_convergence": (
        "wb_dyn_vorticity_flux.yaml",
        ["--LEVEL=850"],
        "era5",
        "aifs_lead",
    ),
    "surface_temperature_energy": (
        "wb_dyn_surface_energy.yaml",
        [],
        "era5_sfc",
        "aifs_sfc_lead",
    ),
}
LEADS = [12, 24, 36, 48]
PRECIP_LEADS = [6, 12, 18, 24, 30, 36, 42, 48]
THRESHOLDS = [5, 10, 20, 50]
WINDOW_KM = 0.25 * 111.2


def bake(recipe, variables, inputs, out_dir):
    """Run one recipe as cset bake does and return its per-case values by valid time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe_vars = parse_variable_options(list(variables), [str(p) for p in inputs])
    execute_recipe(parse_recipe(RECIPES / recipe, recipe_vars), out_dir)
    (nc,) = [f for f in glob.glob(str(out_dir / "*.nc"))]
    cube = iris.load_cube(nc)
    time = cube.coord("time")
    stamps = [f"{t:%Y-%m-%d_%H}" for t in time.units.num2date(time.points)]
    return dict(zip(stamps, map(float, cube.data), strict=True))


def main(out_json):
    """Bake every Dynamic and FSS job and write the results to JSON."""
    results = {"dynamic": {}, "fss_sea_domain": {}}
    for metric, (recipe, variables, truth, model) in DYNAMIC.items():
        for lead in LEADS:
            vals = bake(
                recipe,
                variables,
                [PREP / truth, PREP / f"{model}{lead:02d}"],
                BAKE / f"{metric}_lead{lead:02d}",
            )
            results["dynamic"].setdefault(metric, {})[str(lead)] = vals
            print("dynamic", metric, lead, len(vals), flush=True)
    for lead in PRECIP_LEADS:
        for thr in THRESHOLDS:
            vals = bake(
                "wb_precip_fss.yaml",
                [f"--THRESHOLD={thr}", f"--WINDOW_KM={WINDOW_KM}"],
                [PREP / "gpm", PREP / f"aifs_precip_lead{lead:02d}"],
                BAKE / f"fss_{thr}mm_lead{lead:02d}",
            )
            results["fss_sea_domain"][f"{lead}|{thr}"] = vals
            print("fss", lead, thr, len(vals), flush=True)
    Path(out_json).write_text(json.dumps(results, indent=1))
    print("wrote", out_json)


if __name__ == "__main__":
    main(sys.argv[1])

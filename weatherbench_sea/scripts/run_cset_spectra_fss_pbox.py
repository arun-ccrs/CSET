"""Bake the HKE spectrum recipe and the FSS recipe on PRECIP_DOMAIN for AIFS, 12 cases.

Same mechanism as run_cset_recipes.py (what `cset bake` does). Results go to JSON:
  spectra_by_level_raw[lead][level]  RMSE between ERA5 and AIFS mean spectra
  fss_precip_domain["lead|thr"][case] per-case FSS on 12S-23N, 92-127E

Usage: WB_CSET_WORK=<work dir> HDF5_USE_FILE_LOCKING=FALSE python run_cset_spectra_fss_pbox.py OUT_JSON
"""

import glob
import json
import sys
from pathlib import Path

import iris

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cset_recipes import (
    BAKE,
    PRECIP_LEADS,
    PREP,
    RECIPES,
    THRESHOLDS,
    WINDOW_KM,
    bake,
)

from CSET._common import parse_recipe, parse_variable_options
from CSET.operators import execute_recipe

SPECTRA_LEADS = [12, 24, 36, 48]
SPECTRA_LEVELS = [850, 700, 200]


def bake_scalar(recipe, variables, inputs, out_dir):
    """Run one recipe that produces a single value and return it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe_vars = parse_variable_options(list(variables), [str(p) for p in inputs])
    execute_recipe(parse_recipe(RECIPES / recipe, recipe_vars), out_dir)
    (nc,) = glob.glob(str(out_dir / "*.nc"))
    return float(iris.load_cube(nc).data)


def main(out_json):
    """Bake the spectrum and precipitation-box FSS jobs and write the results to JSON."""
    results = {"spectra_by_level_raw": {}, "fss_precip_domain": {}}
    for lead in SPECTRA_LEADS:
        for level in SPECTRA_LEVELS:
            val = bake_scalar(
                "wb_dyn_hke_spectrum.yaml",
                [f"--LEVEL={level}"],
                [PREP / "era5", PREP / f"aifs_lead{lead:02d}"],
                BAKE / f"hke_spectrum_{level}_lead{lead:02d}",
            )
            results["spectra_by_level_raw"].setdefault(str(lead), {})[str(level)] = val
            print("spectrum", lead, level, val, flush=True)
    for lead in PRECIP_LEADS:
        for thr in THRESHOLDS:
            vals = bake(
                "wb_precip_fss.yaml",
                [f"--THRESHOLD={thr}", f"--WINDOW_KM={WINDOW_KM}"],
                [PREP / "gpm_pbox", PREP / f"aifs_precip_pbox_lead{lead:02d}"],
                BAKE / f"fss_pbox_{thr}mm_lead{lead:02d}",
            )
            results["fss_precip_domain"][f"{lead}|{thr}"] = vals
            print("fss pbox", lead, thr, len(vals), flush=True)
    Path(out_json).write_text(json.dumps(results, indent=1))
    print("wrote", out_json)


if __name__ == "__main__":
    main(sys.argv[1])

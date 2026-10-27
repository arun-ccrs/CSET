"""Reference numbers from the unmodified weatherbench.py, run on the CAWRET files.

Calls weatherbench.py's own per-file functions for the 12 cases:
  - Dynamic (pl + sfc) per case and lead, to check against the shipped npz cache
    and then to compare CSET against;
  - FSS with the original PRECIP_DOMAIN, to check against fss/fss_aifs_12cases.csv;
  - FSS with PRECIP_DOMAIN set to the SEA DOMAIN (the port's box), as CSET's target.

Usage (eccodes needs ECCODES_DIR in the overlay venv):
  ECCODES_DIR=<cset-env> wb-venv/bin/python scripts/wb_reference.py CONFIG OUT_JSON
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

# Data and outputs live in a work directory outside the repo: WB_CSET_WORK, default the current
# directory. It holds prepped/ (from prep_cf.py), reference/ and the bake outputs.
WORK = Path(os.environ.get("WB_CSET_WORK", Path.cwd()))
spec = importlib.util.spec_from_file_location(
    "weatherbench", WORK / "reference" / "weatherbench.py"
)
wb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wb)


def dynamic(config):
    """Per-case Dynamic values from weatherbench.py for the 12 cases."""
    pl_truth, sfc_truth = wb.resolve_pl_truth(config), wb.resolve_sfc_truth(config)
    out = {v: {} for v in wb.DYN_VARS}
    pl_files = wb.filter_truth_files(
        wb.glob_truth_files_by_template(pl_truth.dir, pl_truth.filename_template),
        "12cases",
    )
    for f in pl_files:
        stamp = wb.parse_truth_datetime(f.name).strftime("%Y-%m-%d_%H")
        r = wb.process_dyn_pl_file(f, config)
        for v in ("mse_convergence", "mse_divergence", "vorticity_convergence"):
            for lead, vals in r[v].items():
                if vals:
                    out[v].setdefault(str(lead), {})[stamp] = float(vals[0])
    sfc_files = wb.filter_truth_files(
        wb.glob_truth_files_by_template(sfc_truth.dir, sfc_truth.filename_template),
        "12cases",
    )
    style = wb.infer_sfc_var_style(config)
    for f in sfc_files:
        stamp = wb.parse_truth_datetime(f.name).strftime("%Y-%m-%d_%H")
        r = wb.process_dyn_sfc_file(f, config, sfc_truth, style)
        for lead, vals in r.items():
            if vals:
                out["surface_temperature_energy"].setdefault(str(lead), {})[stamp] = (
                    float(vals[0])
                )
    print("dynamic: pl files", len(pl_files), "sfc files", len(sfc_files), flush=True)
    return out


def fss(config, domain):
    """FSS store from weatherbench.py with PRECIP_DOMAIN set to the given box."""
    wb.PRECIP_DOMAIN = dict(domain)
    store = wb.compute_fss_for_config(config, "12cases")
    return {
        f"{lead}|{thr}|{win}": [float(x) for x in vals]
        for (lead, thr, win), vals in store.items()
    }


def main(config_path, out_path):
    """Run the Dynamic and FSS references and write them to JSON."""
    config = wb.load_model_config(config_path)
    original_precip_domain = dict(wb.PRECIP_DOMAIN)
    result = {
        "dynamic": dynamic(config),
        "fss_precip_domain": fss(config, original_precip_domain),
        "fss_sea_domain": fss(config, wb.DOMAIN),
        "precip_domain": original_precip_domain,
        "sea_domain": dict(wb.DOMAIN),
    }
    Path(out_path).write_text(json.dumps(result, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

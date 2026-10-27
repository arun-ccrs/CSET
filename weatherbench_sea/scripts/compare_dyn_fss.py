"""Compare CSET Dynamic and FSS results with weatherbench.py (live run and shipped files).

Usage: python compare_dyn_fss.py WB_REFERENCE_JSON CSET_JSON
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Data and outputs live in a work directory outside the repo: WB_CSET_WORK, default the current
# directory. It holds prepped/ (from prep_cf.py), reference/ and the bake outputs.
REF = Path(os.environ.get("WB_CSET_WORK", Path.cwd())) / "reference"
DYN_VARS = [
    "mse_convergence",
    "mse_divergence",
    "vorticity_convergence",
    "surface_temperature_energy",
]
LEADS = [12, 24, 36, 48]
TOL = 1e-4


def rel(a, b):
    """Relative difference of a from b, NaN when b is zero."""
    return (a - b) / b if b else np.nan


def main(wb_json, cset_json):
    """Print the Dynamic and FSS comparison tables and a pass/fail summary."""
    wb = json.loads(Path(wb_json).read_text())
    cs = json.loads(Path(cset_json).read_text())
    npz = np.load(REF / "era5_vs_aifs_combined_12cases.npz", allow_pickle=True)[
        "dynamic"
    ].item()
    dyn_ref = np.load(REF / "era5_dynamic_truth_mean.npz", allow_pickle=True)[
        "truth_mean"
    ].item()
    worst = {
        "wb_vs_npz": 0.0,
        "cset_vs_wb": 0.0,
        "fss_wb_vs_csv": 0.0,
        "fss_cset_vs_wb": 0.0,
    }

    print("DYNAMIC, AIFS, 12 cases: mean of per-case RMSE")
    print(
        f"{'metric':28}{'lead':>5}{'shipped npz':>14}{'wb.py live':>14}{'CSET':>14}"
        f"{'wb vs npz':>11}{'CSET vs wb':>11}{'worst case':>11}{'n':>4}"
    )
    for v in DYN_VARS:
        for lead in LEADS:
            n_mean = float(np.mean(npz[v][lead]))
            w = wb["dynamic"][v][str(lead)]
            c = cs["dynamic"][v][str(lead)]
            common = sorted(set(w) & set(c))
            w_mean, c_mean = np.mean(list(w.values())), np.mean(list(c.values()))
            case_worst = max(abs(rel(c[k], w[k])) for k in common)
            sorted_npz_vs_wb = np.max(
                np.abs(np.sort(npz[v][lead]) / np.sort(list(w.values())) - 1)
            )
            worst["wb_vs_npz"] = max(worst["wb_vs_npz"], sorted_npz_vs_wb)
            worst["cset_vs_wb"] = max(
                worst["cset_vs_wb"], case_worst, abs(rel(c_mean, w_mean))
            )
            print(
                f"{v:28}{lead:>5}{n_mean:>14.6g}{w_mean:>14.6g}{c_mean:>14.6g}"
                f"{rel(w_mean, n_mean):>11.1e}{rel(c_mean, w_mean):>11.1e}{case_worst:>11.1e}{len(common):>4}"
            )

    csv = pd.read_csv(REF / "dynamic_12cases.csv")
    aifs = csv[csv["model"].str.startswith("AIFS")]
    print(
        "\nNormalised Dynamic score (CSET mean / shipped ERA5 reference) vs dynamic_12cases.csv, AIFS row"
    )
    for v in DYN_VARS:
        label = {
            "mse_convergence": "MSE Convergence",
            "mse_divergence": "MSE Divergence",
            "vorticity_convergence": "Vorticity Convergence",
            "surface_temperature_energy": "Surface Temperature",
        }[v]
        row = aifs[aifs["variable"].str.contains(label)]
        cset_norm = [
            np.mean(list(cs["dynamic"][v][str(l)].values())) / dyn_ref[v] for l in LEADS
        ]
        shipped = (
            row[[f"lead_{l}h" for l in LEADS]].values[0] if len(row) else [np.nan] * 4
        )
        print(
            f"  {v:28}"
            + "  ".join(
                f"{a:.4f}/{b:.4f}" for a, b in zip(cset_norm, shipped, strict=True)
            )
        )

    print(
        "\nFSS, original precip box: weatherbench.py live vs shipped fss_aifs_12cases.csv"
    )
    shipped = pd.read_csv(REF / "fss_aifs_12cases.csv")
    for key, vals in sorted(
        wb["fss_precip_domain"].items(),
        key=lambda kv: tuple(map(int, kv[0].split("|"))),
    ):
        lead, thr, win = map(int, key.split("|"))
        s = shipped[
            (shipped.lead_time_hr == lead)
            & (shipped.threshold_mm == thr)
            & (shipped.window_gridpts == win)
        ]
        if len(s):
            d = abs(np.mean(vals) - s.fss_avg.values[0])
            worst["fss_wb_vs_csv"] = max(worst["fss_wb_vs_csv"], d)
    print(
        f"  worst |difference| over {len(wb['fss_precip_domain'])} lead/threshold pairs: "
        f"{worst['fss_wb_vs_csv']:.1e} (CSV is rounded to 4 decimals)"
    )

    print("\nFSS, SEA box: CSET vs weatherbench.py (same box), mean over 12 cases")
    print(
        f"{'lead':>5}"
        + "".join(f"{f'{t} mm wb/CSET':>20}" for t in [5, 10, 20, 50])
        + f"{'penalty wb/CSET':>20}"
    )
    for lead in [6, 12, 18, 24, 30, 36, 42, 48]:
        cells, wbm, csm = [], [], []
        for thr in [5, 10, 20, 50]:
            w = np.array(wb["fss_sea_domain"][f"{lead}|{thr}|1"])
            c = np.array(list(cs["fss_sea_domain"][f"{lead}|{thr}"].values()))
            worst["fss_cset_vs_wb"] = max(
                worst["fss_cset_vs_wb"], float(np.max(np.abs(np.sort(w) - np.sort(c))))
            )
            wbm.append(w.mean())
            csm.append(c.mean())
            cells.append(f"{w.mean():.4f}/{c.mean():.4f}")
        print(
            f"{lead:>5}"
            + "".join(f"{x:>20}" for x in cells)
            + f"{f'{1 - np.mean(wbm):.4f}/{1 - np.mean(csm):.4f}':>20}"
        )

    print("\nSummary")
    print(
        f"  Dynamic: weatherbench.py live vs shipped npz, worst relative difference  {worst['wb_vs_npz']:.1e}"
    )
    print(
        f"  Dynamic: CSET vs weatherbench.py, worst relative difference (per case)   {worst['cset_vs_wb']:.1e}"
        f"  -> {'PASS' if worst['cset_vs_wb'] < TOL else 'FAIL'} (threshold {TOL:g})"
    )
    print(
        f"  FSS: weatherbench.py live vs shipped CSV (precip box), worst |difference|  {worst['fss_wb_vs_csv']:.1e}"
    )
    print(
        f"  FSS: CSET vs weatherbench.py (SEA box), worst |difference| per case     {worst['fss_cset_vs_wb']:.1e}"
        f"  -> {'PASS' if worst['fss_cset_vs_wb'] < TOL else 'FAIL'} (threshold {TOL:g})"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

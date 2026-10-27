"""WeatherBench Combined score with AIFS computed by CSET, checked against combined_12cases.csv.

The other four models come from the shipped weatherbench files (npz/, fss/,
reference_values/), exactly as weatherbench.py reads them. AIFS comes from CSET:
Traditional via read_cubes + scores_rmse, Dynamic, HKE spectrum and FSS from the
baked recipe results. The aggregation follows weatherbench.compute_combined_scores:

  base     = 4 Traditional NRMSE + 4 Dynamic normalised scores, per lead
  scale    = max of every model's base values over all leads (global_scale)
  combined = mean(base + [spectra * scale, (1 - mean FSS) * scale])

Usage (wb-venv: imports both CSET and weatherbench.py):
  python combined_score.py WB_REPO PREPPED DYN_FSS_JSON SPECTRA_FSS_PBOX_JSON
"""

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LEADS = [12, 24, 36, 48]
MODELS = ["pangu", "fourcastnet", "aifs", "graphcast", "aurora"]
TRAD = [
    ("z", 500, "geopotential"),
    ("t", 850, "air_temperature"),
    ("q", 700, "specific_humidity"),
    ("wind", 850, None),
]
DYN = [
    "mse_convergence",
    "mse_divergence",
    "vorticity_convergence",
    "surface_temperature_energy",
]


def load_weatherbench(repo):
    """Import weatherbench.py from a checkout so its readers and constants can be used."""
    spec = importlib.util.spec_from_file_location(
        "weatherbench", Path(repo) / "scripts" / "weatherbench.py"
    )
    wb = importlib.util.module_from_spec(spec)
    sys.modules["weatherbench"] = wb
    spec.loader.exec_module(wb)
    return wb


def cset_traditional_raw(prepped, lead):
    """Mean over cases of per-case RMSE, computed with CSET operators."""
    from CSET.operators import constraints, read, scoreswrappers

    def per_case(varname, level):
        c = constraints.combine_constraints(
            variable_constraint=constraints.generate_var_constraint(varname),
            level_constraint=constraints.generate_level_constraint("pressure", level),
        )
        cubes = read.read_cubes(
            [f"{prepped}/era5", f"{prepped}/aifs_lead{lead:02d}"],
            constraint=c,
            model_names=["ERA5", "AIFS"],
        )
        return np.asarray(
            scoreswrappers.scores_rmse(cubes, preserved_coordinates=["time"]).data,
            float,
        )

    raw = {k: per_case(name, lev).mean() for k, lev, name in TRAD if name}
    u, v = per_case("eastward_wind", 850), per_case("northward_wind", 850)
    raw["wind"] = np.mean(np.sqrt(u**2 + v**2))
    return raw


def cset_aifs_metrics(wb, prepped, dyn_json, spec_json):
    """Normalised AIFS components (Traditional, Dynamic, spectrum, FSS penalty) from CSET."""
    trad_ref = wb.get_reference_values(None)
    dyn_ref = wb.get_dynamic_reference_values(None, None)
    spec_ref = wb.get_spectra_reference_values(None)
    dyn = json.loads(Path(dyn_json).read_text())["dynamic"]
    spec = json.loads(Path(spec_json).read_text())
    base, spectra, precip = {}, {}, {}
    for lt in LEADS:
        raw = cset_traditional_raw(prepped, lt)
        for k, _, _ in TRAD:
            base.setdefault(f"trad_{k}", {})[lt] = wb.normalize_trad(
                k, raw[k], trad_ref
            )
        for v in DYN:
            base.setdefault(f"dyn_{v}", {})[lt] = (
                np.mean(list(dyn[v][str(lt)].values())) / dyn_ref[v]
            )
        levels = spec["spectra_by_level_raw"][str(lt)]
        spectra[lt] = float(
            np.mean([levels[str(lev)] / spec_ref[lev] for lev in wb.SPECTRA_PRESSURES])
        )
    for lt in LEADS:
        fss = [
            np.mean(list(spec["fss_precip_domain"][f"{lt}|{thr}"].values()))
            for thr in wb.PRECIP_THRESHOLDS
        ]
        precip[lt] = float(1 - np.mean(fss))
    return {"base": base, "spectra": spectra, "precip": precip}


def shipped_metrics(wb, model):
    """Normalised components for one model from the shipped weatherbench files."""
    trad = wb.get_traditional_data(model, "12cases", {})
    dyn = wb.get_dynamic_data(model, "12cases", {})
    base = {f"trad_{k}": trad[k] for k, _, _ in TRAD}
    base.update({f"dyn_{v}": dyn[v] for v in DYN})
    return {
        "base": base,
        "spectra": wb.get_spectra_data(model, "12cases", {}),
        "precip": wb.get_precip_data(model, "12cases", {}),
    }


def combined(all_metrics):
    """Compute the Combined score per model and lead, and the global_scale used."""
    pool = [
        float(v)
        for m in all_metrics.values()
        for d in m["base"].values()
        for lt in LEADS
        if not np.isnan(v := d.get(lt, np.nan))
    ]
    scale = max(pool)
    out = {}
    for model, m in all_metrics.items():
        out[model] = {}
        for lt in LEADS:
            vals = [
                d[lt] for d in m["base"].values() if not np.isnan(d.get(lt, np.nan))
            ]
            for extra in (m["spectra"].get(lt, np.nan), m["precip"].get(lt, np.nan)):
                if not np.isnan(extra):
                    vals.append(extra * scale)
            out[model][lt] = float(np.mean(vals))
    return out, scale


def main(repo, prepped, dyn_json, spec_json):
    """Print the AIFS components and the Combined table against combined_12cases.csv."""
    wb = load_weatherbench(repo)
    shipped = {m: shipped_metrics(wb, m) for m in MODELS}
    cset = dict(shipped)
    cset["aifs"] = cset_aifs_metrics(wb, prepped, dyn_json, spec_json)

    print("AIFS components, CSET vs shipped (normalised, 12 cases)")
    print(f"{'component':34}" + "".join(f"{f'{lt} h':>22}" for lt in LEADS))
    rows = [
        (k, cset["aifs"]["base"][k], shipped["aifs"]["base"][k])
        for k in shipped["aifs"]["base"]
    ]
    rows += [
        (
            "spectra (mean of 3 levels)",
            cset["aifs"]["spectra"],
            shipped["aifs"]["spectra"],
        ),
        (
            "precip penalty (1 - mean FSS)",
            cset["aifs"]["precip"],
            shipped["aifs"]["precip"],
        ),
    ]
    worst_component = 0.0
    for name, c, s in rows:
        cells = []
        for lt in LEADS:
            rel = c[lt] / s[lt] - 1
            worst_component = max(worst_component, abs(rel))
            cells.append(f"{c[lt]:.5g} ({rel:+.0e})")
        print(f"{name:34}" + "".join(f"{x:>22}" for x in cells))

    csv = pd.read_csv(Path(repo) / "figures" / "combined_12cases.csv").set_index(
        "model"
    )
    ours, scale = combined(cset)
    ship, ship_scale = combined(shipped)
    wb_scores = wb.compute_combined_scores(MODELS, "12cases", {})
    print(f"\nglobal_scale: with CSET AIFS {scale:.6g}, all shipped {ship_scale:.6g}")
    print(
        "\nCombined score, 12 cases: CSV | this aggregation (all shipped) | with CSET AIFS"
    )
    worst_agg = worst_cset = 0.0
    for m in MODELS:
        label = wb.MODEL_DISPLAY.get(m, m)
        cells = []
        for lt in LEADS:
            c = csv.loc[label, f"lead_{lt}h"]
            worst_agg = max(worst_agg, abs(ship[m][lt] - c), abs(wb_scores[m][lt] - c))
            worst_cset = max(worst_cset, abs(ours[m][lt] - c))
            cells.append(f"{c:.4f}|{ship[m][lt]:.4f}|{ours[m][lt]:.4f}")
        print(f"  {label:13}" + "  ".join(cells))
    print(
        f"\nWorst AIFS component relative difference (CSET vs shipped): {worst_component:.1e}"
    )
    print(
        f"Aggregation check (all shipped vs CSV), worst |difference|:  {worst_agg:.1e}"
    )
    print(
        f"With CSET AIFS vs CSV, worst |difference| over all rows:   {worst_cset:.1e}"
        f"  -> {'PASS' if worst_cset < 1e-4 else 'FAIL'} (threshold 1e-4)"
    )


if __name__ == "__main__":
    main(*sys.argv[1:5])

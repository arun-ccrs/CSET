# WeatherBench SEA in CSET

Port of the Southeast Asia regional WeatherBench
([arun-ccrs/weatherbench](https://github.com/arun-ccrs/weatherbench)) into CSET,
and the harness used to check that CSET reproduces its numbers.

## What is in the fork

| Where | What |
|---|---|
| `src/CSET/operators/energetics.py` | Moist static energy, relative vorticity, horizontal flux divergence, surface energy flux divergence, sign masking. WeatherBench constants by default (cp 1004, Lv 2.5e6, Re 6371 km). |
| `src/CSET/operators/scoreswrappers.py` | `scores_fss` (FSS per time, window given in km). `scores_rmse` and friends now also accept fields without a horizontal grid, such as spectra. |
| `src/CSET/operators/power_spectrum.py` | `hke_fft_axis_mean_spectrum`: WeatherBench's HKE spectrum (2D FFT, averaged along one axis). |
| `src/CSET/recipes/weatherbench_sea/` | Recipes: Traditional Z500 RMSE, MSE flux convergence/divergence, vorticity-weighted MSE flux, surface energy, HKE spectrum, FSS. |
| `tests/operators/` | `test_energetics.py`, plus FSS, non-spatial RMSE and spectrum tests. Synthetic data only. |
| `weatherbench_sea/scripts/` | Prep, bake and comparison scripts (below). |
| `weatherbench_sea/results/` | Comparison reports from the AIFS 12-case validation. |

## Validation status (AIFS, 12 cases, CAWRET data)

| Component | CSET vs weatherbench.py, worst difference |
|---|---|
| Traditional NRMSE (Z500, T850, Q700, wind 850), 12-48 h | 2e-6 relative |
| Dynamic: MSE convergence/divergence, vorticity, surface energy | 1e-7 relative, per case |
| HKE spectrum, 850/700/200 hPa | 3e-7 relative |
| FSS, 4 thresholds x 8 leads | identical per case |
| Combined score, all rows of `combined_12cases.csv` | 1.1e-5 (FSS CSVs are rounded to 4 decimals) |

Validated against weatherbench as of October 2026: Traditional and Dynamic on
10S-25N, 90-140E, FSS on 12S-23N, 92-127E (its `PRECIP_DOMAIN`), and the CAWRET
ERA5 surface files. Those surface files are ERA5-Land (land points only) and the
benchmark is moving every metric to the FSS box. Revalidate against the new
weatherbench outputs once that change lands.

## Rerunning the validation

The scripts keep data outside the repo, in a work directory given by
`WB_CSET_WORK` (default: the current directory). It needs:

```
$WB_CSET_WORK/
  prepped/       written by prep_cf.py
  reference/     weatherbench.py and its shipped npz/, reference_values/, figures/ and fss/ files
```

Set `HDF5_USE_FILE_LOCKING=FALSE` when working on `/data`, or NetCDF reads can hang.

1. **Prep** (needs iris and iris-grib; reads CAWRET):
   `python prep_cf.py $WB_CSET_WORK/prepped`, then the same with `sfc_precip` and
   `precip_box` as a second argument. This writes CF NetCDF, clipped to the box,
   with `pressure`, `forecast_reference_time` and `forecast_period`. It fixes four
   loading issues: ERA5 humidity is GRIB param 133, which iris-grib doesn't
   recognise; AIFS uses `level` and labels geopotential as metres; AIFS has no
   start time or lead; GPM has no time or units.
2. **Reference**: `wb_reference.py CONFIG OUT_JSON` runs weatherbench.py's own
   functions on the raw files (configs/aifs_cawret.json).
3. **CSET**: `cset bake` the recipes directly, or `run_cset_recipes.py` and
   `run_cset_spectra_fss_pbox.py` to bake every lead, level and threshold.
4. **Compare**: `cset_rmse.py` (Traditional), `compare_dyn_fss.py`
   (Dynamic, FSS), `combined_score.py` (Combined).

Example, one recipe:

```
cset bake -i $WB_CSET_WORK/prepped/era5 $WB_CSET_WORK/prepped/aifs_lead12 -o out/ \
  -r src/CSET/recipes/weatherbench_sea/wb_dyn_mse_flux.yaml --LEVEL=850 --SIGN=negative
```

## Not done yet

- Workflow integration: a loader, Rose switches, and Combined as a step after baking.
- The other four models (Pangu, FourCastNet, GraphCast, Aurora).
- Revalidation on the single FSS box with ERA5 single-level surface truth.

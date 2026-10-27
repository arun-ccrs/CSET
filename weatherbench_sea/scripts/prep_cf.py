"""Write CF-compliant, domain-clipped NetCDF from the CAWRET ERA5 and AIFS files.

Phase 0 of the WeatherBench -> CSET port: CSET can read the raw files, but
needs CF names, a ``pressure`` coordinate and forecast time metadata to pair
ERA5 with forecasts. This script supplies them for the 12 case studies:
pressure levels (Traditional, Dynamic), 10 m wind and 2 m temperature (Dynamic
surface energy) and precipitation against GPM (FSS). Every field is clipped to
the SEA box, FSS included (user decision; weatherbench.py uses PRECIP_DOMAIN).

Usage: python prep_cf.py OUT_DIR [sfc_precip | precip_box]
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import cf_units
import iris
import iris.coord_systems
import iris.coords
import iris.cube
import iris.util
import numpy as np

warnings.filterwarnings("ignore")

CAWRET = Path("/home/project/13004327/evaluation_service/AIweatherbench")
ERA5_PL = CAWRET / "era5_data/era5_pl_splits_6hr_final"
ERA5_SFC = CAWRET / "era5_data/era5_sfc_splits_6hr_regridded"
ERA5_T2M = CAWRET / "era5_data/era5_sfc_splits_6hr_t2m_regridded"
GPM = CAWRET / "gpm_folder/accumulated"
AIFS = CAWRET / "aifs_folder/aifs_output_regridded"

# Same as weatherbench.py: TWELVE_CASE_DATES, LEADS and DOMAIN.
CASES = [
    "2024-01-24_00",
    "2024-02-06_00",
    "2024-03-06_00",
    "2024-04-12_00",
    "2024-05-04_00",
    "2024-06-21_12",
    "2024-07-12_06",
    "2024-08-09_00",
    "2024-09-17_06",
    "2024-10-14_00",
    "2024-11-16_00",
    "2024-12-23_00",
]
LEADS = [12, 24, 36, 48]
# weatherbench.py TWELVE_CASE_DATES_PRECIP and PRECIP_LEAD_TIMES: FSS uses the precipitation peaks.
PRECIP_CASES = [
    "2024-01-24_12",
    "2024-02-06_12",
    "2024-03-06_12",
    "2024-04-12_06",
    "2024-05-04_06",
    "2024-06-22_00",
    "2024-07-12_06",
    "2024-08-09_06",
    "2024-09-17_12",
    "2024-10-14_06",
    "2024-11-16_06",
    "2024-12-23_12",
]
PRECIP_LEADS = [6, 12, 18, 24, 30, 36, 42, 48]
LAT = (-10.0, 25.0)
LON = (90.0, 140.0)
# weatherbench.py PRECIP_DOMAIN, the FSS box (and the box every metric will use after the 2026 fix).
PRECIP_LAT = (-12.0, 23.0)
PRECIP_LON = (92.0, 127.0)
LEVELS = [200, 500, 700, 850]

# CF standard name and units for each source variable.
ERA5_NAMES = {
    "geopotential": ("geopotential", "m2 s-2"),
    "air_temperature": ("air_temperature", "K"),
    "x_wind": ("eastward_wind", "m s-1"),
    "y_wind": ("northward_wind", "m s-1"),
}
AIFS_NAMES = {
    "z": ("geopotential", "m2 s-2"),  # file says "m" but values are geopotential
    "t": ("air_temperature", "K"),
    "q": ("specific_humidity", "kg kg-1"),
    "u": ("eastward_wind", "m s-1"),
    "v": ("northward_wind", "m s-1"),
}
AIFS_SFC_NAMES = {
    "10u": ("eastward_wind", "m s-1"),
    "10v": ("northward_wind", "m s-1"),
    "2t": ("air_temperature", "K"),
}
PRECIP = ("lwe_thickness_of_precipitation_amount", "mm")
HOURS = "hours since 1970-01-01 00:00:00"
GEOG_CS = iris.coord_systems.GeogCS(6367470.0)


def clip(cube, lat_range=LAT, lon_range=LON):
    """Clip to a WeatherBench box (inclusive edges) with latitude ascending."""
    for old, new in (("lat", "latitude"), ("lon", "longitude")):  # GPM uses lat/lon
        if cube.coords(old):
            cube.coord(old).rename(new)
    cube = cube.extract(
        iris.Constraint(
            latitude=lambda c: lat_range[0] <= c <= lat_range[1],
            longitude=lambda c: lon_range[0] <= c <= lon_range[1],
        )
    )
    lat = cube.coord("latitude")
    if lat.points[0] > lat.points[-1]:
        cube = iris.util.reverse(cube, "latitude")
    # CSET always regrids some fields (e.g. specific humidity) and iris refuses to regrid
    # between a grid with a coordinate system and one without, so match ERA5's (from iris-grib).
    for name in ("latitude", "longitude"):
        coord = cube.coord(name)
        coord.units = "degrees"  # AIFS has none; the iris regridder requires degrees
        if coord.coord_system is None:
            coord.coord_system = GEOG_CS
    return cube


def set_times(cube, init, valid):
    """Replace time metadata with scalar time, forecast_reference_time and forecast_period."""
    for name in ("time", "forecast_reference_time", "forecast_period"):
        if cube.coords(name):
            cube.remove_coord(name)
    unit = cf_units.Unit(HOURS, calendar="standard")
    cube.add_aux_coord(
        iris.coords.AuxCoord(unit.date2num(valid), standard_name="time", units=unit)
    )
    cube.add_aux_coord(
        iris.coords.AuxCoord(
            unit.date2num(init), standard_name="forecast_reference_time", units=unit
        )
    )
    cube.add_aux_coord(
        iris.coords.AuxCoord(
            (valid - init) / timedelta(hours=1),
            standard_name="forecast_period",
            units="hours",
        )
    )
    return cube


def tidy(cube, std_name, units):
    """Set CF name and units, load the data, and name the vertical coordinate pressure."""
    cube.standard_name, cube.long_name, cube.var_name = std_name, None, None
    cube.units = units
    cube.attributes.clear()
    _ = cube.data  # realise: the lazy-save path fails with iris 3.12 + dask 2025.4
    if cube.coords("level"):
        p = cube.coord("level")
        p.rename("pressure")
        p.units = "hPa"
    if cube.coords("pressure"):
        cube.coord("pressure").points = cube.coord("pressure").points.astype(np.float32)
    for name in ("originating_centre",):
        if cube.coords(name):
            cube.remove_coord(name)
    return cube


def era5_cubes(valid):
    """ERA5 pressure-level fields at one valid time, clipped and CF-named."""
    path = ERA5_PL / f"era5_pl_{valid:%Y-%m-%d_%H}.grib"
    out = iris.cube.CubeList()
    for cube in iris.load(path):
        name = cube.name()
        if name in ERA5_NAMES:
            std, units = ERA5_NAMES[name]
        elif getattr(cube.attributes.get("GRIB_PARAM"), "number", None) == 133:
            std, units = (
                "specific_humidity",
                "kg kg-1",
            )  # iris-grib does not map ECMWF param 133
        else:
            raise ValueError(f"Unexpected ERA5 field {name} in {path}")
        cube = tidy(clip(cube), std, units)
        out.append(set_times(cube, valid, valid))
    return out


def aifs_cubes(init, lead):
    """AIFS pressure-level fields for one start time and lead, clipped and CF-named."""
    valid = init + timedelta(hours=lead)
    path = AIFS / f"aifs_gridded_{init:%Y-%m-%d_%H}-out-{lead}_regridded.nc"
    out = iris.cube.CubeList()
    for var, (std, units) in AIFS_NAMES.items():
        cube = iris.load_cube(path, iris.NameConstraint(var_name=var))
        cube = iris.util.squeeze(cube.extract(iris.Constraint(level=LEVELS)))
        cube = tidy(clip(cube), std, units)
        out.append(set_times(cube, init, valid))
    return out


def era5_sfc_cubes(valid):
    """10 m wind and 2 m temperature; t2m lives in its own directory."""
    stamp = f"{valid:%Y-%m-%d_%H}"
    out = iris.cube.CubeList()
    for cube in iris.load(ERA5_SFC / f"era5_sfc_{stamp}_regridded.grib"):
        if cube.name() not in ("x_wind", "y_wind"):
            continue  # also holds total precipitation (param 228), unused here
        std, units = ERA5_NAMES[cube.name()]
        out.append(set_times(tidy(clip(cube), std, units), valid, valid))
    t2m = iris.load_cube(ERA5_T2M / f"era5_sfc_{stamp}_t2m_regridded.grib")
    out.append(set_times(tidy(clip(t2m), "air_temperature", "K"), valid, valid))
    return out


def aifs_path(init, lead):
    """Path of the CAWRET AIFS file for a start time and lead (lead not zero-padded)."""
    return AIFS / f"aifs_gridded_{init:%Y-%m-%d_%H}-out-{lead}_regridded.nc"


def aifs_sfc_cubes(init, lead):
    """AIFS 10 m wind and 2 m temperature for one start time and lead."""
    valid = init + timedelta(hours=lead)
    out = iris.cube.CubeList()
    for var, (std, units) in AIFS_SFC_NAMES.items():
        cube = iris.util.squeeze(
            iris.load_cube(aifs_path(init, lead), iris.NameConstraint(var_name=var))
        )
        out.append(set_times(tidy(clip(cube), std, units), init, valid))
    return out


def gpm_cube(valid, box=(LAT, LON)):
    """GPM accumulated precipitation at one valid time, clipped to the given box."""
    cube = iris.load_cube(GPM / f"gpm_acc_{valid:%Y-%m-%d_%H}.nc")
    return set_times(tidy(clip(cube, *box), *PRECIP), valid, valid)


def aifs_precip_cube(init, lead, box=(LAT, LON)):
    """AIFS tp in metres to mm, negatives clipped to 0, as load_precip_forecast does."""
    cube = iris.util.squeeze(
        iris.load_cube(aifs_path(init, lead), iris.NameConstraint(var_name="tp"))
    )
    cube = clip(cube, *box)
    cube.data = np.clip(cube.data.astype(np.float32) * 1000.0, 0, None)
    return set_times(tidy(cube, *PRECIP), init, init + timedelta(hours=lead))


def main_sfc_precip(out_dir):
    """Write surface fields and SEA-box precipitation for the 12 cases."""
    out_dir = Path(out_dir)
    for case in CASES:
        valid = datetime.strptime(case, "%Y-%m-%d_%H")
        (out_dir / "era5_sfc").mkdir(parents=True, exist_ok=True)
        iris.save(era5_sfc_cubes(valid), out_dir / "era5_sfc" / f"era5_sfc_{case}.nc")
        for lead in LEADS:
            init = valid - timedelta(hours=lead)
            d = out_dir / f"aifs_sfc_lead{lead:02d}"
            d.mkdir(exist_ok=True)
            iris.save(
                aifs_sfc_cubes(init, lead),
                d / f"aifs_sfc_{init:%Y-%m-%d_%H}_lead{lead:02d}.nc",
            )
        print("done sfc", case, flush=True)
    for case in PRECIP_CASES:
        valid = datetime.strptime(case, "%Y-%m-%d_%H")
        (out_dir / "gpm").mkdir(parents=True, exist_ok=True)
        iris.save(gpm_cube(valid), out_dir / "gpm" / f"gpm_{case}.nc")
        for lead in PRECIP_LEADS:
            init = valid - timedelta(hours=lead)
            d = out_dir / f"aifs_precip_lead{lead:02d}"
            d.mkdir(exist_ok=True)
            iris.save(
                aifs_precip_cube(init, lead),
                d / f"aifs_precip_{init:%Y-%m-%d_%H}_lead{lead:02d}.nc",
            )
        print("done precip", case, flush=True)


def main_precip_box(out_dir):
    """GPM and AIFS precipitation on PRECIP_DOMAIN, to reproduce the shipped fss/*.csv."""
    out_dir = Path(out_dir)
    box = (PRECIP_LAT, PRECIP_LON)
    for case in PRECIP_CASES:
        valid = datetime.strptime(case, "%Y-%m-%d_%H")
        (out_dir / "gpm_pbox").mkdir(parents=True, exist_ok=True)
        iris.save(gpm_cube(valid, box), out_dir / "gpm_pbox" / f"gpm_{case}.nc")
        for lead in PRECIP_LEADS:
            init = valid - timedelta(hours=lead)
            d = out_dir / f"aifs_precip_pbox_lead{lead:02d}"
            d.mkdir(exist_ok=True)
            iris.save(
                aifs_precip_cube(init, lead, box),
                d / f"aifs_precip_{init:%Y-%m-%d_%H}_lead{lead:02d}.nc",
            )
        print("done precip box", case, flush=True)


def main(out_dir):
    """Write pressure-level fields for the 12 cases."""
    out_dir = Path(out_dir)
    (out_dir / "era5").mkdir(parents=True, exist_ok=True)
    for case in CASES:
        valid = datetime.strptime(case, "%Y-%m-%d_%H")
        iris.save(era5_cubes(valid), out_dir / "era5" / f"era5_pl_{case}.nc")
        for lead in LEADS:
            init = valid - timedelta(hours=lead)
            d = out_dir / f"aifs_lead{lead:02d}"
            d.mkdir(exist_ok=True)
            iris.save(
                aifs_cubes(init, lead), d / f"aifs_{init:%Y-%m-%d_%H}_lead{lead:02d}.nc"
            )
        print("done", case, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "sfc_precip":
        main_sfc_precip(sys.argv[1])
    elif len(sys.argv) > 2 and sys.argv[2] == "precip_box":
        main_precip_box(sys.argv[1])
    else:
        main(sys.argv[1])

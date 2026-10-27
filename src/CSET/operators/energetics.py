# © Crown copyright, Met Office (2022-2026) and CSET contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Operators for energy and vorticity flux diagnostics on latitude-longitude grids.

These follow the Dynamic benchmark of the Southeast Asia regional WeatherBench
(arun-ccrs/weatherbench). Defaults for the physical constants are the ones that
benchmark uses, so results reproduce it; pass other values to use CSET's.
Arithmetic stays in the input dtype, as in that benchmark.
"""

import iris.cube
import numpy as np

from CSET._common import iter_maybe

# WeatherBench SEA constants (weatherbench.py: CP, G, LV, RE).
CP = 1004.0  # J/kg/K.
LV = 2.5e6  # J/kg.
EARTH_RADIUS = 6371000.0  # m.
SURFACE_AIR_DENSITY = 1.2  # kg/m3.


def _lat_lon_metrics(cube: iris.cube.Cube, earth_radius: float):
    """Return lat/lon axes and the metre spacings used by np.gradient."""
    lat = cube.coord("latitude")
    lon = cube.coord("longitude")
    (lat_dim,) = cube.coord_dims(lat)
    (lon_dim,) = cube.coord_dims(lon)
    lats = lat.points.astype(np.float64)
    lons = lon.points.astype(np.float64)
    dlat = np.deg2rad(np.gradient(lats))
    dlon = np.deg2rad(np.gradient(lons))
    coslat = np.clip(np.cos(np.deg2rad(lats)), 1e-3, 1.0)

    def expand(values, dim):
        shape = [1] * cube.ndim
        shape[dim] = values.size
        return values.reshape(shape)

    dx = expand(dlon, lon_dim) * earth_radius * expand(coslat, lat_dim)
    dy = expand(dlat, lat_dim) * earth_radius
    return lat_dim, lon_dim, dx, dy


def _nan_safe_gradient(data, axis):
    """Gradient with NaNs zero-filled first and restored after, as WeatherBench does."""
    nan_mask = np.isnan(data)
    grad = np.gradient(np.where(nan_mask, 0.0, data), axis=axis)
    grad[nan_mask] = np.nan
    return grad


def _values(cube: iris.cube.Cube):
    """Cube data as a plain array with masked points set to NaN."""
    data = cube.data
    if np.ma.isMaskedArray(data):
        data = np.ma.filled(data.astype(np.result_type(data.dtype, np.float32)), np.nan)
    return data


def _like(template: iris.cube.Cube, data, name: str, units: str) -> iris.cube.Cube:
    """Copy of template with new data, name and units; keeps model attributes."""
    out = template.copy(data=data)
    out.standard_name = None
    out.long_name = name
    out.var_name = name
    out.units = units
    return out


def _single_or_list(cubes: iris.cube.CubeList):
    return cubes[0] if len(cubes) == 1 else cubes


def moist_static_energy(
    temperature: iris.cube.Cube | iris.cube.CubeList,
    geopotential: iris.cube.Cube | iris.cube.CubeList,
    specific_humidity: iris.cube.Cube | iris.cube.CubeList,
    cp: float = CP,
    lv: float = LV,
) -> iris.cube.Cube | iris.cube.CubeList:
    r"""Calculate moist static energy.

    .. math:: h = c_p T + \Phi + L_v q

    Arguments
    ---------
    temperature: iris.cube.Cube | iris.cube.CubeList
        Air temperature in K, one cube per model.
    geopotential: iris.cube.Cube | iris.cube.CubeList
        Geopotential in m2 s-2 (not geopotential height), one cube per model.
    specific_humidity: iris.cube.Cube | iris.cube.CubeList
        Specific humidity in kg kg-1, one cube per model.
    cp: float, optional
        Specific heat capacity of dry air. Default 1004 J/kg/K.
    lv: float, optional
        Latent heat of vaporisation. Default 2.5e6 J/kg.

    Returns
    -------
    iris.cube.Cube | iris.cube.CubeList
        Moist static energy in J kg-1, one cube per model.
    """
    out = iris.cube.CubeList()
    for t, z, q in zip(
        iter_maybe(temperature),
        iter_maybe(geopotential),
        iter_maybe(specific_humidity),
        strict=True,
    ):
        data = cp * _values(t) + _values(z) + lv * _values(q)
        out.append(_like(t, data, "moist_static_energy", "J kg-1"))
    return _single_or_list(out)


def relative_vorticity(
    eastward_wind: iris.cube.Cube | iris.cube.CubeList,
    northward_wind: iris.cube.Cube | iris.cube.CubeList,
    earth_radius: float = EARTH_RADIUS,
) -> iris.cube.Cube | iris.cube.CubeList:
    r"""Calculate relative vorticity with centred differences on a lat-lon grid.

    .. math:: \zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}

    Arguments
    ---------
    eastward_wind, northward_wind: iris.cube.Cube | iris.cube.CubeList
        Wind components in m s-1, one cube per model.
    earth_radius: float, optional
        Default 6371000 m.

    Returns
    -------
    iris.cube.Cube | iris.cube.CubeList
        Relative vorticity in s-1, one cube per model.
    """
    out = iris.cube.CubeList()
    for u, v in zip(iter_maybe(eastward_wind), iter_maybe(northward_wind), strict=True):
        lat_dim, lon_dim, dx, dy = _lat_lon_metrics(u, earth_radius)
        zeta = (
            np.gradient(_values(v), axis=lon_dim) / dx
            - np.gradient(_values(u), axis=lat_dim) / dy
        )
        out.append(_like(u, zeta, "relative_vorticity", "s-1"))
    return _single_or_list(out)


def horizontal_flux_divergence(
    eastward_wind: iris.cube.Cube | iris.cube.CubeList,
    northward_wind: iris.cube.Cube | iris.cube.CubeList,
    scalar: iris.cube.Cube | iris.cube.CubeList,
    multiplier: iris.cube.Cube | iris.cube.CubeList | None = None,
    nan_safe_meridional: bool = False,
    earth_radius: float = EARTH_RADIUS,
) -> iris.cube.Cube | iris.cube.CubeList:
    r"""Calculate the horizontal divergence of a scalar flux on a lat-lon grid.

    .. math:: \nabla \cdot (s m \mathbf{V})
              = \frac{\partial (u s m)}{\partial x} + \frac{\partial (v s m)}{\partial y}

    Negative values are flux convergence. Missing (masked or NaN) points stay
    NaN and make their neighbours' centred differences NaN too.

    Arguments
    ---------
    eastward_wind, northward_wind: iris.cube.Cube | iris.cube.CubeList
        Wind components in m s-1, one cube per model.
    scalar: iris.cube.Cube | iris.cube.CubeList
        The transported quantity, for example moist static energy.
    multiplier: iris.cube.Cube | iris.cube.CubeList, optional
        A second factor in the flux, for example relative vorticity to get the
        vorticity-weighted MSE flux.
    nan_safe_meridional: bool, optional
        Zero-fill NaNs before the meridional difference and restore them after,
        so neighbours of a missing point keep a value. The WeatherBench SEA
        benchmark does this for the vorticity-weighted MSE flux only. Default False.
    earth_radius: float, optional
        Default 6371000 m.

    Returns
    -------
    iris.cube.Cube | iris.cube.CubeList
        Flux divergence, one cube per model.
    """
    multipliers = (
        iter_maybe(multiplier)
        if multiplier is not None
        else [None] * len(iter_maybe(scalar))
    )
    out = iris.cube.CubeList()
    for u, v, s, m in zip(
        iter_maybe(eastward_wind),
        iter_maybe(northward_wind),
        iter_maybe(scalar),
        multipliers,
        strict=True,
    ):
        lat_dim, lon_dim, dx, dy = _lat_lon_metrics(s, earth_radius)
        weight = _values(s)
        units = s.units * u.units / "m"
        if m is not None:
            weight = weight * _values(m)
            units = units * m.units
        zonal = np.gradient(_values(u) * weight, axis=lon_dim) / dx
        meridional_flux = _values(v) * weight
        if nan_safe_meridional:
            meridional = _nan_safe_gradient(meridional_flux, lat_dim) / dy
        else:
            meridional = np.gradient(meridional_flux, axis=lat_dim) / dy
        out.append(
            _like(s, zonal + meridional, f"flux_divergence_of_{s.name()}", str(units))
        )
    return _single_or_list(out)


def surface_energy_flux_divergence(
    eastward_wind: iris.cube.Cube | iris.cube.CubeList,
    northward_wind: iris.cube.Cube | iris.cube.CubeList,
    temperature: iris.cube.Cube | iris.cube.CubeList,
    cp: float = CP,
    air_density: float = SURFACE_AIR_DENSITY,
    earth_radius: float = EARTH_RADIUS,
) -> iris.cube.Cube | iris.cube.CubeList:
    r"""Calculate the divergence of the near-surface sensible heat flux.

    .. math:: \nabla \cdot (\rho c_p T_{2m} \mathbf{V}_{10m})

    Arguments
    ---------
    eastward_wind, northward_wind: iris.cube.Cube | iris.cube.CubeList
        10 m wind components in m s-1, one cube per model.
    temperature: iris.cube.Cube | iris.cube.CubeList
        2 m air temperature in K, one cube per model.
    cp: float, optional
        Default 1004 J/kg/K.
    air_density: float, optional
        Constant near-surface air density. Default 1.2 kg m-3.
    earth_radius: float, optional
        Default 6371000 m.

    Returns
    -------
    iris.cube.Cube | iris.cube.CubeList
        Energy flux divergence in W m-3, one cube per model.
    """
    out = iris.cube.CubeList()
    for u, v, t in zip(
        iter_maybe(eastward_wind),
        iter_maybe(northward_wind),
        iter_maybe(temperature),
        strict=True,
    ):
        energy = _like(
            t, cp * _values(t) * air_density, "near_surface_heat_content", "J m-3"
        )
        div = horizontal_flux_divergence(u, v, energy, earth_radius=earth_radius)
        div.long_name = div.var_name = "surface_energy_flux_divergence"
        out.append(div)
    return _single_or_list(out)


def keep_sign(
    cubes: iris.cube.Cube | iris.cube.CubeList, sign: str = "negative"
) -> iris.cube.Cube | iris.cube.CubeList:
    """Set values of the other sign (and zero) to NaN.

    Use ``sign="negative"`` to keep flux convergence and ``"positive"`` to keep
    divergence. Scores then compare only points where both fields share the sign.

    Arguments
    ---------
    cubes: iris.cube.Cube | iris.cube.CubeList
        Fields to mask.
    sign: str
        "negative" or "positive".

    Returns
    -------
    iris.cube.Cube | iris.cube.CubeList
        Masked copies, with the suffix "_convergence" or "_divergence" on the name.
    """
    if sign not in ("negative", "positive"):
        raise ValueError(f"sign must be 'negative' or 'positive', not {sign!r}")
    out = iris.cube.CubeList()
    for cube in iter_maybe(cubes):
        data = _values(cube)
        keep = data < 0 if sign == "negative" else data > 0
        suffix = "_convergence" if sign == "negative" else "_divergence"
        out.append(
            _like(
                cube,
                np.where(keep, data, np.nan),
                cube.name() + suffix,
                str(cube.units),
            )
        )
    return _single_or_list(out)

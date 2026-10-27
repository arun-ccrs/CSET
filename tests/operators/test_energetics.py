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

"""Tests for energetics operators."""

import iris.coords
import iris.cube
import numpy as np
import pytest

from CSET.operators import energetics

LATS = np.arange(-10.0, 10.01, 0.25)
LONS = np.arange(100.0, 115.01, 0.25)


def _cube(data, name="x", units="1", model="M"):
    cube = iris.cube.Cube(
        np.asanyarray(data),  # keep masks
        long_name=name,
        units=units,
        dim_coords_and_dims=[
            (iris.coords.DimCoord(LATS, standard_name="latitude", units="degrees"), 0),
            (iris.coords.DimCoord(LONS, standard_name="longitude", units="degrees"), 1),
        ],
    )
    cube.attributes["model_name"] = model
    return cube


def _random(seed, scale=1.0, offset=0.0):
    rng = np.random.default_rng(seed)
    return offset + scale * rng.standard_normal((LATS.size, LONS.size))


def _weatherbench_reference(u, v, mse):
    """compute_mse_vorticity_convergence from weatherbench.py, verbatim maths."""
    re = 6371000
    dlat = np.deg2rad(np.gradient(LATS))
    dlon = np.deg2rad(np.gradient(LONS))
    coslat = np.clip(np.cos(np.deg2rad(LATS)), 1e-3, 1.0)
    zeta = np.gradient(v, axis=-1) / (
        dlon[None, :] * re * coslat[:, None]
    ) - np.gradient(u, axis=-2) / (dlat[:, None] * re)
    mse_div = np.gradient(u * mse, axis=-1) / (
        dlon[None, :] * re * coslat[:, None]
    ) + np.gradient(v * mse, axis=-2) / (dlat[:, None] * re)
    nan_mask = np.isnan(v * zeta * mse)
    zeta_dvdy = np.gradient(np.where(nan_mask, 0.0, v * zeta * mse), axis=-2) / (
        dlat[:, None] * re
    )
    zeta_dvdy[nan_mask] = np.nan
    zeta_div = (
        np.gradient(u * mse * zeta, axis=-1) / (dlon[None, :] * re * coslat[:, None])
        + zeta_dvdy
    )
    return zeta, mse_div, zeta_div


def test_moist_static_energy_formula():
    """Moist static energy is cp T + z + Lv q with the WeatherBench constants."""
    t, z, q = (
        _cube(np.full((81, 61), 300.0)),
        _cube(np.full((81, 61), 1.5e4)),
        _cube(np.full((81, 61), 0.01)),
    )
    mse = energetics.moist_static_energy(t, z, q)
    np.testing.assert_allclose(mse.data, 1004.0 * 300.0 + 1.5e4 + 2.5e6 * 0.01)
    assert mse.units == "J kg-1"
    assert mse.attributes["model_name"] == "M"


def test_moist_static_energy_cubelist_pairs_models():
    """Each model's cubes are combined with each other, in order."""
    t = iris.cube.CubeList(
        [
            _cube(np.full((81, 61), 300.0), model="A"),
            _cube(np.full((81, 61), 290.0), model="B"),
        ]
    )
    z = iris.cube.CubeList(
        [_cube(np.zeros((81, 61)), model="A"), _cube(np.zeros((81, 61)), model="B")]
    )
    q = iris.cube.CubeList(
        [_cube(np.zeros((81, 61)), model="A"), _cube(np.zeros((81, 61)), model="B")]
    )
    out = energetics.moist_static_energy(t, z, q)
    assert [c.attributes["model_name"] for c in out] == ["A", "B"]
    np.testing.assert_allclose(out[1].data, 1004.0 * 290.0)


def test_relative_vorticity_of_zonal_shear():
    """Zonal wind growing linearly northward gives constant vorticity -du/dy."""
    dy = np.deg2rad(0.25) * energetics.EARTH_RADIUS
    shear = 1e-5
    u = _cube(
        np.repeat((np.arange(LATS.size) * dy * shear)[:, None], LONS.size, axis=1)
    )
    v = _cube(np.zeros((LATS.size, LONS.size)))
    zeta = energetics.relative_vorticity(u, v)
    np.testing.assert_allclose(zeta.data, -shear, rtol=1e-10)


def test_flux_divergence_of_uniform_flow_is_zero():
    """A uniform flux has zero divergence."""
    out = energetics.horizontal_flux_divergence(
        _cube(np.full((81, 61), 5.0)),
        _cube(np.full((81, 61), -3.0)),
        _cube(np.full((81, 61), 2.0)),
    )
    np.testing.assert_allclose(out.data, 0.0, atol=1e-20)


def test_matches_weatherbench_formulas():
    """MSE flux divergence and vorticity-weighted flux divergence equal weatherbench.py's."""
    u, v = _random(1, 8.0), _random(2, 8.0)
    t, z, q = _random(3, 2.0, 290.0), _random(4, 50.0, 1.5e4), _random(5, 1e-3, 0.012)
    mse_ref = 1004 * t + 9.81 * (z / 9.81) + 2.5e6 * q
    zeta_ref, div_ref, zdiv_ref = _weatherbench_reference(u, v, mse_ref)

    cu, cv = _cube(u, units="m s-1"), _cube(v, units="m s-1")
    mse = energetics.moist_static_energy(
        _cube(t, units="K"), _cube(z, units="m2 s-2"), _cube(q, units="kg kg-1")
    )
    zeta = energetics.relative_vorticity(cu, cv)
    div = energetics.horizontal_flux_divergence(cu, cv, mse)
    zdiv = energetics.horizontal_flux_divergence(
        cu, cv, mse, multiplier=zeta, nan_safe_meridional=True
    )

    np.testing.assert_allclose(zeta.data, zeta_ref, rtol=1e-12)
    np.testing.assert_allclose(
        div.data, div_ref, rtol=1e-9, atol=1e-12 * np.abs(div_ref).max()
    )
    np.testing.assert_allclose(
        zdiv.data, zdiv_ref, rtol=1e-9, atol=1e-12 * np.abs(zdiv_ref).max()
    )


def test_masked_points_become_nan_not_fill_values():
    """Masked input must not leak its fill value into the gradients."""
    t = np.ma.masked_array(np.full((81, 61), 300.0), mask=False)
    t.mask[40, 30] = True
    t.data[40, 30] = 9.97e36
    out = energetics.surface_energy_flux_divergence(
        _cube(np.full((81, 61), 1.0)),
        _cube(np.full((81, 61), 1.0)),
        _cube(t, units="K"),
    )
    # Centred differences at a point use its neighbours, so the neighbours become NaN.
    for neighbour in [(39, 30), (41, 30), (40, 29), (40, 31)]:
        assert np.isnan(out.data[neighbour])
    assert np.nanmax(np.abs(out.data)) < 1.0


def test_surface_energy_matches_weatherbench():
    """div(rho cp T2m V10) with rho = 1.2 and cp = 1004, as compute_t2m_energy."""
    u10, v10, t2m = _random(6, 5.0), _random(7, 5.0), _random(8, 1.5, 300.0)
    dlat, dlon = np.deg2rad(np.gradient(LATS)), np.deg2rad(np.gradient(LONS))
    coslat = np.clip(np.cos(np.deg2rad(LATS)), 1e-3, 1.0)
    e = 1004 * t2m * 1.2
    ref = np.gradient(u10 * e, axis=-1) / (
        dlon[None, :] * 6371000 * coslat[:, None]
    ) + np.gradient(v10 * e, axis=-2) / (dlat[:, None] * 6371000)
    out = energetics.surface_energy_flux_divergence(
        _cube(u10), _cube(v10), _cube(t2m, units="K")
    )
    np.testing.assert_allclose(out.data, ref, rtol=1e-9, atol=1e-12 * np.abs(ref).max())


@pytest.mark.parametrize(("sign", "kept"), [("negative", [-1.0]), ("positive", [2.0])])
def test_keep_sign(sign, kept):
    """Only values of the requested sign are kept; the rest become NaN."""
    data = np.zeros((81, 61))
    data[0, 0], data[0, 1] = -1.0, 2.0
    out = energetics.keep_sign(_cube(data), sign)
    np.testing.assert_array_equal(out.data[~np.isnan(out.data)], kept)


def test_keep_sign_rejects_unknown_sign():
    """An unknown sign raises a ValueError."""
    with pytest.raises(ValueError):
        energetics.keep_sign(_cube(np.zeros((81, 61))), "both")

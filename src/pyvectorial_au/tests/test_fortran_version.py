import os
import pytest
import astropy.units as u
import numpy as np
from pyvectorial.backends.fortran_version import vmr_from_fortran_output


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def fortran_fort16_file(rootdir):
    return os.path.join(rootdir, "input/fort.16")


@pytest.fixture
def vmc_from_fortran(fortran_fort16_file):
    return vmr_from_fortran_output(fortran_fort16_file, read_sputter=True)


def test_fortran_collision_sphere(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.collision_sphere_radius, 0.9e7 * u.cm)


def test_fortran_max_grid_radius(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.max_grid_radius, 1.061e6 * u.km)  # type: ignore


def test_fortran_coma_radius(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.coma_radius, 7.259e5 * u.km)  # type: ignore


# def test_num_fragments_theory(vmc_from_fortran):
#     assert np.isclose(vmc_from_fortran.num_fragments_theory, 0.545e33)
#
#
# def test_num_fragments_grid(vmc_from_fortran):
#     assert np.isclose(vmc_from_fortran.num_fragments_grid, 0.533e33)


def test_volume_density_grid_first(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.volume_density_grid[0], 0.2e3 * u.km)  # type: ignore


def test_volume_density_grid_last(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.volume_density_grid[-1], 0.10e7 * u.km)  # type: ignore


def test_volume_density_first(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.volume_density[0], 0.34e5 / u.cm**3)


def test_volume_density_last(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.volume_density[-1], 0.15e-3 / u.cm**3)


def test_column_density_grid_first(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.column_density_grid[0], 0.20322e3 * u.km)  # type: ignore


def test_column_density_grid_last(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.column_density_grid[-1], 7.21038e5 * u.km)  # type: ignore


def test_column_density_first(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.column_density[0], 7.73995e12 / u.cm**2)


def test_column_density_last(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.column_density[-1], 4.88174e8 / u.cm**2)


def test_fragment_sputter_r_first(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.fragment_sputter.rs[0], 1048352.88 * u.km)  # type: ignore


def test_fragment_sputter_r_last(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.fragment_sputter.rs[-1], 203.169159 * u.km)  # type: ignore


def test_fragment_sputter_theta_first(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.fragment_sputter.thetas[0], 6.04152456e-2)


def test_fragment_sputter_theta_last(vmc_from_fortran):
    assert np.isclose(vmc_from_fortran.fragment_sputter.thetas[-1], 3.08117747)


def test_fragment_sputter_density_first(vmc_from_fortran):
    assert np.isclose(
        vmc_from_fortran.fragment_sputter.fragment_density[0], 6.28462021e-6 / u.cm**3
    )


def test_fragment_sputter_density_last(vmc_from_fortran):
    assert np.isclose(
        vmc_from_fortran.fragment_sputter.fragment_density[-1], 3.87345695 / u.cm**3
    )

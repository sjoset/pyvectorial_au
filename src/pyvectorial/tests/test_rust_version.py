import os
import pytest
import astropy.units as u
import numpy as np
from pyvectorial.backends.rust_version import vmc_from_rust_output, vmr_from_rust_output


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def rust_output_file(rootdir):
    return os.path.join(rootdir, "input/rust_out.txt")


@pytest.fixture
def vmc_from_rust(rust_output_file):
    return vmc_from_rust_output(rust_output_file)


@pytest.fixture
def vmr_from_rust(rust_output_file):
    return vmr_from_rust_output(
        rust_output_file, vmc_from_rust_output(rust_output_file)
    )


def test_rust_collision_sphere(vmr_from_rust):
    assert np.isclose(vmr_from_rust.collision_sphere_radius, 8.82353e1 * u.km)  # type: ignore


def test_rust_max_grid_radius(vmr_from_rust):
    assert np.isclose(vmr_from_rust.max_grid_radius, 1.87195e6 * u.km)  # type: ignore


def test_rust_coma_radius(vmr_from_rust):
    assert np.isclose(vmr_from_rust.coma_radius, 5.58084e5 * u.km)  # type: ignore


def test_volume_density_grid_first(vmr_from_rust):
    assert np.isclose(vmr_from_rust.volume_density_grid[0], 1.0 * u.km)  # type: ignore


def test_volume_density_grid_last(vmr_from_rust):
    assert np.isclose(vmr_from_rust.volume_density_grid[-1], 1.87195e6 * u.km)  # type: ignore


def test_volume_density_first(vmr_from_rust):
    # assert np.isclose(vmr_from_rust.volume_density[0].to_value(1 / u.m**3), (1.62622935 / u.m**3).to_value(1 / u.m**3))  # type: ignore
    assert np.isclose(vmr_from_rust.volume_density[0], 1.62622935e13 / u.m**3)  # type: ignore


def test_volume_density_last(vmr_from_rust):
    assert np.isclose(vmr_from_rust.volume_density[-1], 7.213311e1 / u.m**3)  # type: ignore


def test_fragment_sputter_r_first(vmr_from_rust):
    assert np.isclose(vmr_from_rust.fragment_sputter.rs[0], 1.0 * u.km)  # type: ignore


def test_fragment_sputter_r_last(vmr_from_rust):
    assert np.isclose(vmr_from_rust.fragment_sputter.rs[-1], 1.8719500607868414372205734e6 * u.km)  # type: ignore


def test_fragment_sputter_theta_first(vmr_from_rust):
    assert np.isclose(
        vmr_from_rust.fragment_sputter.thetas[0], 1.9634954084936206974987272e-2
    )


def test_fragment_sputter_theta_last(vmr_from_rust):
    assert np.isclose(
        vmr_from_rust.fragment_sputter.thetas[-1], 3.1219576995048567980006737e0
    )


def test_fragment_sputter_density_first(vmr_from_rust):
    assert np.isclose(
        vmr_from_rust.fragment_sputter.fragment_density[0], 5.0652281546602539062500000e12 / u.m**3  # type: ignore
    )


def test_fragment_sputter_density_last(vmr_from_rust):
    assert np.isclose(
        vmr_from_rust.fragment_sputter.fragment_density[-1], 2.3438452027657809202163594e-24 / u.m**3  # type: ignore
    )

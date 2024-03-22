import os
import pytest
import astropy.units as u
from pyvectorial_au.model_input.vectorial_model_config_reader import (
    vectorial_model_config_from_yaml,
)


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def single_run_yaml(rootdir):
    return os.path.join(rootdir, "input/single_run.yaml")


@pytest.fixture
def vmc_set_single(single_run_yaml):
    return vectorial_model_config_from_yaml(single_run_yaml)


def test_single_run_base_q(vmc_set_single):
    assert vmc_set_single.production.base_q == (1.0e28 / u.s)


def test_single_run_parent_outflow(vmc_set_single):
    assert vmc_set_single.parent.v_outflow == (0.85 * u.km / u.s)  # type: ignore


def test_single_run_parent_tau_T(vmc_set_single):
    assert vmc_set_single.parent.tau_T == (45000 * u.s)


def test_single_run_parent_tau_d(vmc_set_single):
    assert vmc_set_single.parent.tau_d == (50000 * u.s)


def test_single_run_parent_sigma(vmc_set_single):
    assert vmc_set_single.parent.sigma == (3.0e-16 * u.cm**2)  # type: ignore


def test_single_run_fragment_v_photo(vmc_set_single):
    assert vmc_set_single.fragment.v_photo == (1.05 * u.km / u.s)  # type: ignore


def test_single_run_fragment_tau_T(vmc_set_single):
    assert vmc_set_single.fragment.tau_T == (129000 * u.s)


def test_single_run_grid_radial_points(vmc_set_single):
    assert vmc_set_single.grid.radial_points == 50


def test_single_run_grid_angular_points(vmc_set_single):
    assert vmc_set_single.grid.angular_points == 30


def test_single_run_grid_radial_substeps(vmc_set_single):
    assert vmc_set_single.grid.radial_substeps == 40

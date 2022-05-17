
import os
import pytest
import astropy.units as u
from ..vmconfigread import vm_configs_from_yaml


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def single_run_yaml(rootdir):
    return os.path.join(rootdir, 'input/single_run.yaml')


@pytest.fixture
def eight_run_yaml(rootdir):
    return os.path.join(rootdir, 'input/eight_runs.yaml')


@pytest.fixture
def vmc_set_single(single_run_yaml):
    return vm_configs_from_yaml(single_run_yaml)


@pytest.fixture
def vmc_set_eight(eight_run_yaml):
    return vm_configs_from_yaml(eight_run_yaml)


@pytest.fixture
def vmc_set_eight_prodsorted(eight_run_yaml):
    return sorted(vm_configs_from_yaml(eight_run_yaml), key=lambda x: x.production.base_q)


@pytest.fixture
def vmc_set_eight_ptaud_sorted(eight_run_yaml):
    return sorted(vm_configs_from_yaml(eight_run_yaml), key=lambda x: x.parent.tau_d)


@pytest.fixture
def vmc_set_eight_ftauT_sorted(eight_run_yaml):
    return sorted(vm_configs_from_yaml(eight_run_yaml), key=lambda x: x.fragment.tau_T)


def test_single_run_length(vmc_set_single):

    assert len(vmc_set_single) == 1


def test_single_run_base_q(vmc_set_single):

    assert vmc_set_single[0].production.base_q == (1.e+28 / u.s)


def test_single_run_parent_name(vmc_set_single):

    assert vmc_set_single[0].parent.name == "H2O"


def test_single_run_parent_outflow(vmc_set_single):

    assert vmc_set_single[0].parent.v_outflow == (0.85 * u.km/u.s)


def test_single_run_parent_tau_d(vmc_set_single):

    assert vmc_set_single[0].parent.tau_d == (50000 * u.s)


def test_single_run_parent_ratio(vmc_set_single):

    assert vmc_set_single[0].parent.T_to_d_ratio == 0.93


def test_single_run_parent_sigma(vmc_set_single):

    assert vmc_set_single[0].parent.sigma == (3.0e-16 * u.cm**2)


def test_single_run_fragment_name(vmc_set_single):

    assert vmc_set_single[0].fragment.name == "OH"


def test_single_run_fragment_v_photo(vmc_set_single):

    assert vmc_set_single[0].fragment.v_photo == (1.05 * u.km/u.s)


def test_single_run_fragment_tau_T(vmc_set_single):

    assert vmc_set_single[0].fragment.tau_T == (129000 * u.s)


def test_single_run_comet_name(vmc_set_single):

    assert vmc_set_single[0].comet.name == "Comet Test"


def test_single_run_comet_rh(vmc_set_single):

    assert vmc_set_single[0].comet.rh == (1.4 * u.AU)


def test_single_run_grid_radial_points(vmc_set_single):

    assert vmc_set_single[0].grid.radial_points == 50


def test_single_run_grid_angular_points(vmc_set_single):

    assert vmc_set_single[0].grid.angular_points == 30


def test_single_run_grid_radial_substeps(vmc_set_single):

    assert vmc_set_single[0].grid.radial_substeps == 40


def test_single_run_etc_print(vmc_set_single):

    assert vmc_set_single[0].etc['print_progress'] == False


def test_eight_run_length(vmc_set_eight):

    assert len(vmc_set_eight) == 8


def test_eight_run_base_q(vmc_set_eight_prodsorted):

    assert vmc_set_eight_prodsorted[-1].production.base_q == (1.e+28 / u.s)


def test_eight_run_parent_tau_d(vmc_set_eight_ptaud_sorted):

    assert vmc_set_eight_ptaud_sorted[-1].parent.tau_d == (70000 * u.s)


def test_eight_run_fragment_tau_T(vmc_set_eight_ftauT_sorted):

    assert vmc_set_eight_ftauT_sorted[-1].fragment.tau_T == (100000 * u.s)

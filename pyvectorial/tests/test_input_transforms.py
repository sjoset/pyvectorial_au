
import copy
import pytest
import astropy.units as u
import numpy as np

from ..vmconfig import VectorialModelConfig, Production, Parent, Fragment, Comet, Grid
from ..input_transforms import apply_input_transform, unapply_input_transform


@pytest.fixture
def vmc_two_au() -> VectorialModelConfig:
    """
        Return a valid VectorialModelConfig with comet at 2.0 AU for
        testing input transformations
    """

    q = Production(base_q=1e28, time_variation_type=None, params=None)
    p = Parent(name='parent test', v_outflow=1.0*u.km/u.s, tau_d=100000*u.s,
            tau_T=93000*u.s, T_to_d_ratio=0.93, sigma=3.0e-16*u.cm**2)
    f = Fragment(name='fragment test', v_photo=1.05*u.km/u.s, tau_T=100000*u.s)
    c = Comet(name='comet test', rh=2.0 * u.AU, delta=1.0 * u.AU,
            transform_method=None, transform_applied=False)
    g = Grid(
            radial_points=50,
            angular_points=50,
            radial_substeps=50
            )

    return VectorialModelConfig(
            production=q,
            parent=p,
            fragment=f,
            comet=c,
            grid=g,
            etc=None
            )


def test_festou_fortran_apply(vmc_two_au):

    rhsq = (vmc_two_au.comet.rh.value)**2

    vmc_two_au.comet.transform_method = 'festou_fortran'
    vmc_orig = copy.deepcopy(vmc_two_au)

    apply_input_transform(vmc_two_au)

    transformed_input = [vmc_two_au.parent.tau_d.to(u.s).value,
            vmc_two_au.parent.tau_T.to(u.s).value,
            vmc_two_au.fragment.tau_T.to(u.s).value]
    expected_result = [vmc_orig.parent.tau_d.to(u.s).value*rhsq,
            vmc_orig.parent.tau_T.to(u.s).value*rhsq,
            vmc_orig.fragment.tau_T.to(u.s).value*rhsq]

    assert np.allclose(transformed_input, expected_result)


def test_festou_fortran_unapply(vmc_two_au):

    vmc_two_au.comet.transform_method = 'festou_fortran'
    vmc_orig = copy.deepcopy(vmc_two_au)

    apply_input_transform(vmc_two_au)
    unapply_input_transform(vmc_two_au)

    transformed_input = [vmc_two_au.parent.tau_d.to(u.s).value,
            vmc_two_au.parent.tau_T.to(u.s).value,
            vmc_two_au.fragment.tau_T.to(u.s).value]
    expected_result = [vmc_orig.parent.tau_d.to(u.s).value,
            vmc_orig.parent.tau_T.to(u.s).value,
            vmc_orig.fragment.tau_T.to(u.s).value]

    assert np.allclose(transformed_input, expected_result)


def test_cochran_schleicher_93_apply(vmc_two_au):

    rhsq = (vmc_two_au.comet.rh.value)**2
    sqrh = np.sqrt(vmc_two_au.comet.rh.value)

    vmc_two_au.comet.transform_method = 'cochran_schleicher_93'
    vmc_orig = copy.deepcopy(vmc_two_au)

    apply_input_transform(vmc_two_au)

    transformed_input = [vmc_two_au.parent.tau_d.to(u.s).value,
            vmc_two_au.parent.tau_T.to(u.s).value,
            vmc_two_au.fragment.tau_T.to(u.s).value,
            vmc_two_au.parent.v_outflow.to(u.km/u.s).value]
    expected_result = [vmc_orig.parent.tau_d.to(u.s).value*rhsq,
            vmc_orig.parent.tau_T.to(u.s).value*rhsq,
            vmc_orig.fragment.tau_T.to(u.s).value*rhsq,
            (0.85 * u.km/u.s).value/sqrh]

    assert np.allclose(transformed_input, expected_result)


def test_cochran_schleicher_93_unapply(vmc_two_au):

    sqrh = np.sqrt(vmc_two_au.comet.rh.value)

    vmc_two_au.comet.transform_method = 'cochran_schleicher_93'
    vmc_orig = copy.deepcopy(vmc_two_au)

    apply_input_transform(vmc_two_au)
    unapply_input_transform(vmc_two_au)

    transformed_input = [vmc_two_au.parent.tau_d.to(u.s).value,
            vmc_two_au.parent.tau_T.to(u.s).value,
            vmc_two_au.fragment.tau_T.to(u.s).value,
            vmc_two_au.parent.v_outflow.to(u.km/u.s).value]
    expected_result = [vmc_orig.parent.tau_d.to(u.s).value,
            vmc_orig.parent.tau_T.to(u.s).value,
            vmc_orig.fragment.tau_T.to(u.s).value,
            (0.85 * u.km/u.s).value/sqrh]

    assert np.allclose(transformed_input, expected_result)

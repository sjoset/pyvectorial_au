import pytest
import astropy.units as u
import numpy as np

from ..vectorial_model_config import (
    VectorialModelConfig,
    Production,
    Parent,
    Fragment,
    Grid,
)
from ..input_transforms import (
    apply_input_transform,
    unapply_input_transform,
    VmcTransform,
)


@pytest.fixture
def testing_vmc() -> VectorialModelConfig:
    """
    Return a valid VectorialModelConfig for testing input transformations
    """

    q = Production(base_q=1e28 / u.s)
    p = Parent(
        v_outflow=1.0 * u.km / u.s,
        tau_d=100000 * u.s,
        tau_T=93000 * u.s,
        sigma=3.0e-16 * u.cm**2,
    )
    f = Fragment(v_photo=1.05 * u.km / u.s, tau_T=100000 * u.s)
    g = Grid(
        radial_points=50,
        angular_points=50,
        radial_substeps=50,
        parent_destruction_level=0.99,
        fragment_destruction_level=0.95,
    )

    return VectorialModelConfig(production=q, parent=p, fragment=f, grid=g)


def test_fortran_festou_apply(testing_vmc):
    r_h = 2.0 * u.AU
    rhsq = (r_h.to_value(u.AU)) ** 2

    xfrmed_vmc = apply_input_transform(testing_vmc, r_h, VmcTransform.fortran_festou)
    assert xfrmed_vmc is not None

    transformed_input = [
        xfrmed_vmc.parent.tau_d.to(u.s).value,
        xfrmed_vmc.parent.tau_T.to(u.s).value,
        xfrmed_vmc.fragment.tau_T.to(u.s).value,
    ]
    expected_result = [
        testing_vmc.parent.tau_d.to(u.s).value * rhsq,
        testing_vmc.parent.tau_T.to(u.s).value * rhsq,
        testing_vmc.fragment.tau_T.to(u.s).value * rhsq,
    ]

    assert np.allclose(transformed_input, expected_result)


def test_festou_fortran_unapply(testing_vmc):
    r_h = 2.0 * u.AU

    xfrmed_vmc = apply_input_transform(testing_vmc, r_h, VmcTransform.fortran_festou)
    assert xfrmed_vmc is not None

    unxfrmed_vmc = unapply_input_transform(xfrmed_vmc, r_h, VmcTransform.fortran_festou)
    assert unxfrmed_vmc is not None

    transformed_input = [
        unxfrmed_vmc.parent.tau_d.to(u.s).value,
        unxfrmed_vmc.parent.tau_T.to(u.s).value,
        unxfrmed_vmc.fragment.tau_T.to(u.s).value,
    ]
    expected_result = [
        testing_vmc.parent.tau_d.to(u.s).value,
        testing_vmc.parent.tau_T.to(u.s).value,
        testing_vmc.fragment.tau_T.to(u.s).value,
    ]

    assert np.allclose(transformed_input, expected_result)


def test_cochran_schleicher_93_apply(testing_vmc):
    r_h = 2.0 * u.AU
    rhsq = (r_h.to_value(u.AU)) ** 2
    sqrh = np.sqrt(r_h.to_value(u.AU))

    xfrmed_vmc = apply_input_transform(
        testing_vmc, r_h, VmcTransform.cochran_schleicher_93
    )
    assert xfrmed_vmc is not None

    transformed_input = [
        xfrmed_vmc.parent.tau_d.to(u.s).value,
        xfrmed_vmc.parent.tau_T.to(u.s).value,
        xfrmed_vmc.fragment.tau_T.to(u.s).value,
        xfrmed_vmc.parent.v_outflow.to(u.km / u.s).value,
    ]
    expected_result = [
        testing_vmc.parent.tau_d.to(u.s).value * rhsq,
        testing_vmc.parent.tau_T.to(u.s).value * rhsq,
        testing_vmc.fragment.tau_T.to(u.s).value * rhsq,
        (0.85 * u.km / u.s).value / sqrh,
    ]

    assert np.allclose(transformed_input, expected_result)


def test_cochran_schleicher_93_unapply(testing_vmc):
    r_h = 2.0 * u.AU
    sqrh = np.sqrt(r_h.to_value(u.AU))

    xfrmed_vmc = apply_input_transform(
        testing_vmc, r_h, VmcTransform.cochran_schleicher_93
    )
    assert xfrmed_vmc is not None
    unxfrmed_vmc = unapply_input_transform(
        xfrmed_vmc, r_h, VmcTransform.cochran_schleicher_93
    )
    assert unxfrmed_vmc is not None

    transformed_input = [
        unxfrmed_vmc.parent.tau_d.to(u.s).value,
        unxfrmed_vmc.parent.tau_T.to(u.s).value,
        unxfrmed_vmc.fragment.tau_T.to(u.s).value,
        unxfrmed_vmc.parent.v_outflow.to(u.km / u.s).value,
    ]
    expected_result = [
        testing_vmc.parent.tau_d.to(u.s).value,
        testing_vmc.parent.tau_T.to(u.s).value,
        testing_vmc.fragment.tau_T.to(u.s).value,
        (0.85 * u.km / u.s).value / sqrh,
    ]

    assert np.allclose(transformed_input, expected_result)

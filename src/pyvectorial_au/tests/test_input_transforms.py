import pytest
import astropy.units as u
import numpy as np

from pyvectorial_au.model_input.vectorial_model_config import (
    VectorialModelConfig,
    CometProduction,
    ParentMolecule,
    FragmentMolecule,
    VectorialModelGrid,
)
from pyvectorial_au.pre_model_processing.input_transforms import (
    apply_input_transform,
    VmcTransform,
)


@pytest.fixture
def testing_vmc() -> VectorialModelConfig:
    """
    Return a valid VectorialModelConfig for testing input transformations
    """

    q = CometProduction(base_q_per_s=1e28)
    p = ParentMolecule(
        v_outflow_kms=1.0,
        tau_d_s=100000,
        tau_T_s=93000,
        sigma_cm_sq=3.0e-16,
    )
    f = FragmentMolecule(v_photo_kms=1.05, tau_T_s=100000)
    g = VectorialModelGrid(
        radial_points=50,
        angular_points=50,
        radial_substeps=50,
        parent_destruction_level=0.99,
        fragment_destruction_level=0.95,
    )

    return VectorialModelConfig(production=q, parent=p, fragment=f, grid=g)


def test_fortran_festou_apply(testing_vmc):
    r_h = 2.0 * u.AU  # type: ignore
    r_h_au = r_h.to_value(u.AU)
    rhsq = r_h_au**2

    xfrmed_vmc = apply_input_transform(
        vmc=testing_vmc, r_h=r_h, xfrm=VmcTransform.fortran_festou
    )
    assert xfrmed_vmc is not None

    transformed_input = [
        xfrmed_vmc.parent.tau_d.to_value(u.s),  # type: ignore
        xfrmed_vmc.parent.tau_T.to_value(u.s),  # type: ignore
        xfrmed_vmc.fragment.tau_T.to_value(u.s),  # type: ignore
    ]
    expected_result = [
        testing_vmc.parent.tau_d.to_value(u.s) * rhsq,
        testing_vmc.parent.tau_T.to_value(u.s) * rhsq,
        testing_vmc.fragment.tau_T.to_value(u.s) * rhsq,
    ]

    assert np.allclose(transformed_input, expected_result)  # type: ignore


def test_cochran_schleicher_93_apply(testing_vmc):
    r_h = 2.0 * u.AU  # type: ignore
    r_h_au = r_h.to_value(u.AU)
    rhsq = r_h_au**2
    sqrh = np.sqrt(r_h_au)

    xfrmed_vmc = apply_input_transform(
        testing_vmc, r_h, VmcTransform.cochran_schleicher_93
    )
    assert xfrmed_vmc is not None

    transformed_input = [
        xfrmed_vmc.parent.tau_d.to_value(u.s),  # type: ignore
        xfrmed_vmc.parent.tau_T.to_value(u.s),  # type: ignore
        xfrmed_vmc.fragment.tau_T.to_value(u.s),  # type: ignore
        xfrmed_vmc.parent.v_outflow.to_value(u.km / u.s),  # type: ignore
    ]
    expected_result = [
        testing_vmc.parent.tau_d.to(u.s).value * rhsq,
        testing_vmc.parent.tau_T.to(u.s).value * rhsq,
        testing_vmc.fragment.tau_T.to(u.s).value * rhsq,
        0.85 / sqrh,
    ]

    assert np.allclose(transformed_input, expected_result)  # type: ignore

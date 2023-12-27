#!/usr/bin/env python3

import numpy as np
import astropy.units as u
import sbpy.activity as sba
from astropy import constants as const

from pyvectorial.backends.python_version import PythonModelExtraConfig
from pyvectorial.input_transforms import VmcTransform, apply_input_transform
from pyvectorial.vectorial_model_config import (
    CometProduction,
    FragmentMolecule,
    ParentMolecule,
    VectorialModelConfig,
    VectorialModelGrid,
)
from pyvectorial.vectorial_model_runner import run_vectorial_model


def test_festou92() -> None:
    """Compare to Festou et al. 1992 production rates of comet 6P/d'Arrest.
    Festou et al. 1992. The Gas Production Rate of Periodic Comet d'Arrest.
    Asteroids, Comets, Meteors 1991, 177.
    https://ui.adsabs.harvard.edu/abs/1992acm..proc..177F/abstract
    IUE observations of OH(0-0) band.
    The table appears to have a typo in the Q column for the 1.32 au
    observation: using 2.025 instead of 3.025.
    """

    # Table 2
    r_hs = [1.2912, 1.2949, 1.3089, 1.3200, 1.3366] * u.au  # type: ignore
    deltas = [0.7410, 0.7651, 0.8083, 0.8353, 0.8720] * u.au  # type: ignore
    fluxes = [337e-14, 280e-14, 480e-14, 522e-14, 560e-14] * u.erg / u.cm**2 / u.s  # type: ignore
    gs = [2.33e-4, 2.60e-4, 3.36e-4, 3.73e-4, 4.03e-4] / u.s
    published_Qs = [1.451e28, 1.228e28, 1.967e28, 2.025e28, 2.035e28] / u.s

    # OH (0-0) luminosity per molecule
    L_N = gs / (r_hs / u.au) ** 2 * const.h * const.c / (3086 * u.AA)  # type: ignore

    dummy_input_q = 1e29 / u.s
    # dummy value for running the model
    production = CometProduction(base_q_per_s=dummy_input_q.to_value(1 / u.s))
    # Parent molecule is H2O
    parent = ParentMolecule(
        tau_d_s=72500, tau_T_s=65000, v_outflow_kms=0.85, sigma_cm_sq=3e-16
    )
    # Fragment molecule is OH
    fragment = FragmentMolecule(tau_T_s=160000, v_photo_kms=1.05)
    grid = VectorialModelGrid(
        radial_points=150,
        angular_points=100,
        radial_substeps=80,
        parent_destruction_level=0.99,
        fragment_destruction_level=0.95,
    )
    base_vmc = VectorialModelConfig(
        production=production, parent=parent, fragment=fragment, grid=grid
    )

    # https://pds.nasa.gov/ds-view/pds/viewInstrumentProfile.jsp?INSTRUMENT_ID=LWP&INSTRUMENT_HOST_ID=IUE
    # Large-Aperture Length(arcsec)   22.51+/-0.40
    # Large-Aperture Width(arcsec)     9.91+/-0.17
    #
    # 10x20 quoted by Festou et al.
    # effective circle is 8.0 radius
    # half geometric mean is 7.07
    # lwp = core.CircularAperture(7.07 * u.arcsec)  # geometric mean
    lwp = sba.RectangularAperture((20, 10) * u.arcsec)  # type: ignore
    # lwp = sba.RectangularAperture((9.1, 15.3) * u.arcsec)

    vmrs = [
        run_vectorial_model(
            apply_input_transform(
                vmc=base_vmc, r_h=r_h, xfrm=VmcTransform.fortran_festou
            ),
            extra_config=PythonModelExtraConfig(print_progress=False),
        )
        for r_h in r_hs
    ]
    fragment_counts = [
        vmr.coma.total_number(lwp, eph=delta)
        for vmr, delta in zip(vmrs, deltas)
        if vmr.coma is not None
    ]

    N = u.Quantity(fragment_counts)
    Q_model = (dummy_input_q * fluxes / (L_N * N) * 4 * np.pi * deltas**2).to(
        published_Qs.unit
    )
    Q_ratio = published_Qs / Q_model
    print(f"Q: {published_Qs}")
    print(f"Q_model: {Q_model}")
    print(f"Q/Q_model: {Q_ratio}")

    assert u.allclose(published_Qs, Q_model)


if __name__ == "__main__":
    test_festou92()

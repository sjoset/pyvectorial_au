import logging as log
from enum import Enum, auto

import numpy as np
import astropy.units as u
from astropy.units.quantity import Quantity

from pyvectorial_au.model_input.vectorial_model_config import (
    FragmentMolecule,
    ParentMolecule,
    VectorialModelConfig,
)


"""
    There are different approaches to scaling parameters with heliocentric distance.
    These functions provide some of the common ones dealt with during testing of the vectorial model
"""


class VmcTransform(Enum):
    cochran_schleicher_93 = auto()
    fortran_festou = auto()


def apply_input_transform(
    vmc: VectorialModelConfig, r_h: Quantity, xfrm: VmcTransform
) -> VectorialModelConfig:
    if xfrm == VmcTransform.cochran_schleicher_93:
        return _apply_cochran_schleicher_93(vmc, r_h)
    elif xfrm == VmcTransform.fortran_festou:
        return _apply_festou_fortran(vmc, r_h)


def _apply_cochran_schleicher_93(
    vmc_orig: VectorialModelConfig,
    r_h: Quantity,
) -> VectorialModelConfig:
    rh = r_h.to_value(u.AU)
    rhsquared = rh**2  # type: ignore
    sqrh = np.sqrt(rh)

    new_parent = ParentMolecule(
        v_outflow_kms=0.85 / sqrh,  # type: ignore
        tau_d_s=vmc_orig.parent.tau_d_s * rhsquared,
        tau_T_s=vmc_orig.parent.tau_T_s * rhsquared,
        sigma_cm_sq=vmc_orig.parent.sigma_cm_sq,
    )
    new_fragment = FragmentMolecule(
        tau_T_s=vmc_orig.fragment.tau_T_s * rhsquared,
        v_photo_kms=vmc_orig.fragment.v_photo_kms,
    )

    vmc = vmc_orig.model_copy(
        update={"parent": new_parent, "fragment": new_fragment}, deep=True
    )

    log.debug("Applying transform cochran_schleicher_93...")
    log.debug("Effect of transform at %s AU:", rh)
    log.debug(
        "Parent outflow: %s --> %s", vmc_orig.parent.v_outflow, vmc.parent.v_outflow
    )
    log.debug("Parent tau_d: %s --> %s", vmc_orig.parent.tau_d, vmc.parent.tau_d)
    log.debug("Parent tau_T: %s --> %s", vmc_orig.parent.tau_T, vmc.parent.tau_T)
    log.debug("Fragment tau_T: %s --> %s", vmc_orig.fragment.tau_T, vmc.fragment.tau_T)

    return vmc


def _apply_festou_fortran(
    vmc_orig: VectorialModelConfig, r_h: Quantity
) -> VectorialModelConfig:
    rh = r_h.to(u.AU).value
    rhsquared = rh**2

    new_parent = vmc_orig.parent.model_copy(
        update={
            "tau_d_s": vmc_orig.parent.tau_d_s * rhsquared,
            "tau_T_s": vmc_orig.parent.tau_T_s * rhsquared,
        },
        deep=True,
    )
    new_fragment = vmc_orig.fragment.model_copy(
        update={"tau_T_s": vmc_orig.fragment.tau_T_s * rhsquared}
    )

    vmc = vmc_orig.model_copy(
        update={"parent": new_parent, "fragment": new_fragment}, deep=True
    )

    log.debug("Applying transform fortran_festou...")
    log.debug("Effect of transform at %s AU:", rh)
    log.debug("\tParent tau_d: %s --> %s", vmc_orig.parent.tau_d, vmc.parent.tau_d)
    log.debug("\tParent tau_T: %s --> %s", vmc_orig.parent.tau_T, vmc.parent.tau_T)
    log.debug(
        "\tFragment tau_T: %s --> %s", vmc_orig.fragment.tau_T, vmc.fragment.tau_T
    )

    return vmc

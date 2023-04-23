import copy

import logging as log
import astropy.units as u
from astropy.units.quantity import Quantity
import numpy as np

from enum import Enum, auto
from typing import Optional

from .vectorial_model_config import VectorialModelConfig


"""
    There are different approaches to scaling parameters with heliocentric distance.

    These functions provide some of the common ones dealt with during testing of the vectorial model, as well as
    functions to undo the scaling to recover the original set of parameters.

    The given VectorialModelConfig is modified in-place, a new copy is not returned
"""


class VmcTransform(Enum):
    cochran_schleicher_93 = auto()
    fortran_festou = auto()


def apply_input_transform(
    vmc: VectorialModelConfig, r_h: Quantity, xfrm: VmcTransform
) -> Optional[VectorialModelConfig]:
    if xfrm == VmcTransform.cochran_schleicher_93:
        return _apply_cochran_schleicher_93(vmc, r_h)
    elif xfrm == VmcTransform.fortran_festou:
        return _apply_festou_fortran(vmc, r_h)
    else:
        log.info("Invalid transform method specified, skipping")
        return None


def unapply_input_transform(
    vmc: VectorialModelConfig, r_h: Quantity, xfrm: VmcTransform
) -> Optional[VectorialModelConfig]:
    if xfrm == VmcTransform.cochran_schleicher_93:
        return _unapply_cochran_schleicher_93(vmc, r_h)
    elif xfrm == VmcTransform.fortran_festou:
        return _unapply_festou_fortran(vmc, r_h)
    else:
        log.info("Invalid transform method specified, skipping")
        return None


def _apply_cochran_schleicher_93(
    vmc_orig: VectorialModelConfig,
    r_h: Quantity,
) -> VectorialModelConfig:
    vmc = copy.deepcopy(vmc_orig)

    log.debug("Reminder: cochran_schleicher_93 overwrites v_outflow of parent")
    rh = r_h.to(u.AU).value
    sqrh = np.sqrt(rh)

    v_old = copy.deepcopy(vmc.parent.v_outflow)
    tau_d_old = copy.deepcopy(vmc.parent.tau_d)
    tau_T_old = copy.deepcopy(vmc.parent.tau_T)
    tau_T_old_frag = copy.deepcopy(vmc.fragment.tau_T)

    vmc.parent.v_outflow = (0.85 / sqrh) * u.km / u.s
    vmc.parent.tau_d *= rh**2
    vmc.parent.tau_T *= rh**2
    vmc.fragment.tau_T *= rh**2

    log.debug("Effect of transform at %s AU:", rh)
    log.debug("Parent outflow: %s --> %s", v_old, vmc.parent.v_outflow)
    log.debug("Parent tau_d: %s --> %s", tau_d_old, vmc.parent.tau_d)
    log.debug("Parent tau_T: %s --> %s", tau_T_old, vmc.parent.tau_T)
    log.debug("Fragment tau_T: %s --> %s", tau_T_old_frag, vmc.fragment.tau_T)

    return vmc


def _unapply_cochran_schleicher_93(
    vmc_orig: VectorialModelConfig, r_h: Quantity
) -> VectorialModelConfig:
    vmc = copy.deepcopy(vmc_orig)

    log.debug(
        "Unapplying cochran_schleicher_93 cannot recover original v_outflow of parent"
    )
    rh = r_h.to(u.AU).value

    tau_d_old = copy.deepcopy(vmc.parent.tau_d)
    tau_T_old = copy.deepcopy(vmc.parent.tau_T)
    tau_T_old_frag = copy.deepcopy(vmc.fragment.tau_T)

    vmc.parent.tau_d /= rh**2
    vmc.parent.tau_T /= rh**2
    vmc.fragment.tau_T /= rh**2

    log.debug("Effect of transform at %s AU:", rh)
    log.debug("Parent tau_d: %s --> %s", tau_d_old, vmc.parent.tau_d)
    log.debug("Parent tau_T: %s --> %s", tau_T_old, vmc.parent.tau_T)
    log.debug("Fragment tau_T: %s --> %s", tau_T_old_frag, vmc.fragment.tau_T)

    return vmc


def _apply_festou_fortran(
    vmc_orig: VectorialModelConfig, r_h: Quantity
) -> VectorialModelConfig:
    vmc = copy.deepcopy(vmc_orig)

    rh = r_h.to(u.AU).value
    ptau_d_old = copy.deepcopy(vmc.parent.tau_d)
    ptau_T_old = copy.deepcopy(vmc.parent.tau_T)
    ftau_T_old = copy.deepcopy(vmc.fragment.tau_T)
    vmc.parent.tau_d *= rh**2
    vmc.parent.tau_T *= rh**2
    vmc.fragment.tau_T *= rh**2
    log.debug("\tParent tau_d: %s --> %s", ptau_d_old, vmc.parent.tau_d)
    log.debug("\tParent tau_T: %s --> %s", ptau_T_old, vmc.parent.tau_T)
    log.debug("\tFragment tau_T: %s --> %s", ftau_T_old, vmc.fragment.tau_T)

    return vmc


def _unapply_festou_fortran(
    vmc_orig: VectorialModelConfig, r_h: Quantity
) -> VectorialModelConfig:
    vmc = copy.deepcopy(vmc_orig)

    rh = r_h.to(u.AU).value
    ptau_d_old = copy.deepcopy(vmc.parent.tau_d)
    ptau_T_old = copy.deepcopy(vmc.parent.tau_T)
    ftau_T_old = copy.deepcopy(vmc.fragment.tau_T)
    vmc.parent.tau_d /= rh**2
    vmc.parent.tau_T /= rh**2
    vmc.fragment.tau_T /= rh**2
    log.debug("\tParent tau_d: %s --> %s", ptau_d_old, vmc.parent.tau_d)
    log.debug("\tParent tau_T: %s --> %s", ptau_T_old, vmc.parent.tau_T)
    log.debug("\tFragment tau_T: %s --> %s", ftau_T_old, vmc.fragment.tau_T)

    return vmc

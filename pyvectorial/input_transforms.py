import copy

import logging as log
import astropy.units as u
import numpy as np

from .vmconfig import VectorialModelConfig


"""
    Different papers and researchers take different approaches to scaling parameters with heliocentric distance.

    These functions provide some of the common ones dealt with during testing of the vectorial model, as well as
    functions to undo the scaling to recover the original set of parameters.

    The given VectorialModelConfig is modified in-place, a new copy is not returned
"""


def apply_input_transform(vmc: VectorialModelConfig) -> None:

    log.debug("Current transform state: %s", vmc.comet.transform_applied)

    if vmc.comet.transform_applied:
        log.debug("Attempted to apply transform more than once, skipping")
        return

    # if none specified, nothing to do
    if vmc.comet.transform_method is None:
        log.debug("No valid tranformation of input data specified, no transform applied")
        return

    log.info("Transforming input parameters using method %s", vmc.comet.transform_method)

    if vmc.comet.transform_method == 'cochran_schleicher_93':
        _apply_cochran_schleicher_93(vmc)
    elif vmc.comet.transform_method == 'festou_fortran':
        _apply_festou_fortran(vmc)
    else:
        log.info("Invalid transform method specified, skipping")
        return

    vmc.comet.transform_applied = True


def unapply_input_transform(vmc: VectorialModelConfig) -> None:

    log.debug("Current transform state: %s", vmc.comet.transform_applied)

    if not vmc.comet.transform_applied:
        log.debug("Attempted to unapply transform that has not been applied, skipping")
        return

    # if none specified, nothing to do
    if vmc.comet.transform_method is None:
        log.debug("No valid tranformation of input data specified, no transform unapplied")
        return

    log.info("Untransforming input parameters using method %s", vmc.comet.transform_method)

    if vmc.comet.transform_method == 'cochran_schleicher_93':
        _unapply_cochran_schleicher_93(vmc)
    elif vmc.comet.transform_method == 'festou_fortran':
        _unapply_festou_fortran(vmc)
    else:
        log.info("Invalid transform method specified, skipping")
        return

    vmc.comet.transform_applied = False


def _apply_cochran_schleicher_93(vmc: VectorialModelConfig) -> None:

        log.info("Reminder: cochran_schleicher_93 overwrites v_outflow of parent")
        rh = vmc.comet.rh.to(u.AU).value
        sqrh = np.sqrt(rh)

        v_old = copy.deepcopy(vmc.parent.v_outflow)
        tau_d_old = copy.deepcopy(vmc.parent.tau_d)
        tau_T_old = copy.deepcopy(vmc.parent.tau_T)
        tau_T_old_frag = copy.deepcopy(vmc.fragment.tau_T)

        vmc.parent.v_outflow = (0.85/sqrh) * u.km/u.s
        vmc.parent.tau_d *= rh**2
        vmc.parent.tau_T = vmc.parent.tau_d * vmc.parent.T_to_d_ratio
        vmc.fragment.tau_T *= rh**2

        log.info("Effect of transform at %s AU:", rh)
        log.info("Parent outflow: %s --> %s", v_old, vmc.parent.v_outflow)
        log.info("Parent tau_d: %s --> %s", tau_d_old, vmc.parent.tau_d)
        log.info("Parent tau_T: %s --> %s", tau_T_old, vmc.parent.tau_T)
        log.info("Fragment tau_T: %s --> %s", tau_T_old_frag, vmc.fragment.tau_T)


def _unapply_cochran_schleicher_93(vmc: VectorialModelConfig) -> None:

        log.info("Unapplying cochran_schleicher_93 cannot recover original v_outflow of parent")
        rh = vmc.comet.rh.to(u.AU).value

        tau_d_old = copy.deepcopy(vmc.parent.tau_d)
        tau_T_old = copy.deepcopy(vmc.parent.tau_T)
        tau_T_old_frag = copy.deepcopy(vmc.fragment.tau_T)

        vmc.parent.tau_d /= rh**2
        vmc.parent.tau_T = vmc.parent.tau_d * vmc.parent.T_to_d_ratio
        vmc.fragment.tau_T /= rh**2

        log.info("Effect of transform at %s AU:", rh)
        log.info("Parent tau_d: %s --> %s", tau_d_old, vmc.parent.tau_d)
        log.info("Parent tau_T: %s --> %s", tau_T_old, vmc.parent.tau_T)
        log.info("Fragment tau_T: %s --> %s", tau_T_old_frag, vmc.fragment.tau_T)


def _apply_festou_fortran(vmc: VectorialModelConfig) -> None:

        rh = vmc.comet.rh.to(u.AU).value
        ptau_d_old = copy.deepcopy(vmc.parent.tau_d)
        ptau_T_old = copy.deepcopy(vmc.parent.tau_T)
        ftau_T_old = copy.deepcopy(vmc.fragment.tau_T)
        vmc.parent.tau_d *= rh**2
        vmc.parent.tau_T *= rh**2
        vmc.fragment.tau_T *= rh**2
        log.info("\tParent tau_d: %s --> %s", ptau_d_old, vmc.parent.tau_d)
        log.info("\tParent tau_T: %s --> %s", ptau_T_old, vmc.parent.tau_T)
        log.info("\tFragment tau_T: %s --> %s", ftau_T_old, vmc.fragment.tau_T)


def _unapply_festou_fortran(vmc: VectorialModelConfig) -> None:

        rh = vmc.comet.rh.to(u.AU).value
        ptau_d_old = copy.deepcopy(vmc.parent.tau_d)
        ptau_T_old = copy.deepcopy(vmc.parent.tau_T)
        ftau_T_old = copy.deepcopy(vmc.fragment.tau_T)
        vmc.parent.tau_d /= rh**2
        vmc.parent.tau_T /= rh**2
        vmc.fragment.tau_T /= rh**2
        log.info("\tParent tau_d: %s --> %s", ptau_d_old, vmc.parent.tau_d)
        log.info("\tParent tau_T: %s --> %s", ptau_T_old, vmc.parent.tau_T)
        log.info("\tFragment tau_T: %s --> %s", ftau_T_old, vmc.fragment.tau_T)

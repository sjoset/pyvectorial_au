import hashlib
import pprint

import astropy.units as u

from dataclasses import dataclass
from typing import Optional


"""
    Main VectorialModelConfig dataclass to hold the input parameters that are handed off to sbpy's
    VectorialModel or Festou's original Fortran

    Built from smaller component dataclasses that specify information about the parents, fragments,
    outflow, etc.
"""


@dataclass
class Production:
    base_q: u.quantity.Quantity
    time_variation_type: Optional[str] = None
    # TODO: introduce types for each supported time-variation type
    params: Optional[dict] = None


@dataclass
class Parent:
    tau_d: u.quantity.Quantity
    tau_T: u.quantity.Quantity
    v_outflow: u.quantity.Quantity
    sigma: u.quantity.Quantity


@dataclass
class Fragment:
    v_photo: u.quantity.Quantity
    tau_T: u.quantity.Quantity


@dataclass
class Grid:
    radial_points: int
    angular_points: int
    radial_substeps: int
    # TODO: once the code is working, should these have defaults?
    parent_destruction_level: float
    fragment_destruction_level: float


@dataclass
class VectorialModelConfig:
    production: Production
    parent: Parent
    fragment: Fragment
    grid: Grid


# Our function for generating a deterministic hash based on the vmc
def hash_vmc(vmc: VectorialModelConfig):
    """
    If we have a large number of models stored, we may want to quickly search whether or not a given
    VectorialModelConfig has been run already.  If we generate a unique hash based on the
    VectorialModelConfig and save that along with the model results, we can compare hashes to see if
    this particular VectorialModelConfig has been run already.
    """
    return hashlib.sha256(pprint.pformat(vmc).encode("utf-8")).hexdigest()

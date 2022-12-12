
import astropy.units as u

from dataclasses import dataclass


"""
    Main VectorialModelConfig dataclass to hold the input parameters that are handed off to sbpy's
    VectorialModel or Festou's original Fortran

    Built from smaller component dataclasses that specify information about the parents, fragments,
    outflow, etc.
"""


@dataclass
class Production:
    base_q: u.quantity.Quantity
    time_variation_type: str
    params: dict


@dataclass
class Parent:
    name: str
    v_outflow: u.quantity.Quantity
    tau_d: u.quantity.Quantity
    tau_T: u.quantity.Quantity
    sigma: u.quantity.Quantity
    T_to_d_ratio: float


@dataclass
class Fragment:
    name: str
    v_photo: u.quantity.Quantity
    tau_T: u.quantity.Quantity


@dataclass
class Comet:
    name: str
    rh: u.quantity.Quantity
    delta: u.quantity.Quantity
    transform_method: str
    transform_applied: bool


@dataclass
class Grid:
    radial_points: int
    angular_points: int
    radial_substeps: int


@dataclass
class VectorialModelConfig:
    production: Production
    parent: Parent
    fragment: Fragment
    comet: Comet
    grid: Grid
    etc: dict

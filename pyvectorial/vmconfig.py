
from dataclasses import dataclass


@dataclass
class Production:
    base_q: float
    time_variation_type: str
    params: dict


@dataclass
class Parent:
    name: str
    v_outflow: float
    tau_d: float
    tau_T: float
    sigma: float
    T_to_d_ratio: float


@dataclass
class Fragment:
    name: str
    v_photo: float
    tau_T: float


@dataclass
class Comet:
    name: str
    rh: float
    delta: float
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

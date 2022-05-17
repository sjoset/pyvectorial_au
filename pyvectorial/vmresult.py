
import numpy as np

from dataclasses import dataclass
from scipy.interpolate import PPoly
from astropy.units.quantity import Quantity


@dataclass
class VectorialModelResult:
    volume_density_grid: np.ndarray
    volume_density: np.ndarray
    column_density_grid: np.ndarray
    column_density: np.ndarray
    volume_density_interpolation: PPoly
    column_density_interpolation: PPoly

    num_fragments_theory: float
    num_fragments_grid: float

    max_grid_radius: Quantity

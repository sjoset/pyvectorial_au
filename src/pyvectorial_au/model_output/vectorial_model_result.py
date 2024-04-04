import numpy as np
import sbpy.activity as sba

from dataclasses import dataclass
from scipy.interpolate import PPoly
from astropy.units.quantity import Quantity
from typing import Optional

from pyvectorial_au.model_output.fragment_sputter import FragmentSputterSpherical

"""
Defines VectorialModelResult, a dataclass for holding the result of a vectorial model calculation,
namely, the volume density and fragment sputter.  Functions written to handle VectorialModelResult
don't need to care which model generated the results.
"""


@dataclass
class VectorialModelResult:
    """
    Holds a standardized model result to abstract away the underlying details so that the python,
    fortran, and rust models can be run as backends interchangeably
    """

    """The model calculates the volume density along gridded radial values"""
    volume_density_grid: np.ndarray
    volume_density: np.ndarray

    """ The fragment sputter that produces the volume density above """
    fragment_sputter: FragmentSputterSpherical

    """ Some physical quantities of interest """
    collision_sphere_radius: Quantity
    max_grid_radius: Quantity
    coma_radius: Quantity

    column_density_grid: Optional[np.ndarray] = None
    column_density: Optional[np.ndarray] = None

    volume_density_interpolation: Optional[PPoly] = None
    column_density_interpolation: Optional[PPoly] = None

    coma: Optional[sba.VectorialModel] = None

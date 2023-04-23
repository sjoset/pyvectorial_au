import numpy as np
import sbpy.activity as sba

from dataclasses import dataclass
from scipy.interpolate import PPoly
from astropy.units.quantity import Quantity
from typing import Optional, Union

"""
Defines VectorialModelResult, a dataclass for holding the result of a vectorial model calculation,
namely, the volume density and fragment sputter.  Functions written to handle VectorialModelResult
don't need to care which model generated the results.
"""


# Dataclasses for the two-dimensional fragment sputter information obtained from the model
@dataclass
class FragmentSputterSpherical:
    rs: np.ndarray
    thetas: np.ndarray
    fragment_density: np.ndarray


@dataclass
class FragmentSputterPolar:
    rs: np.ndarray
    thetas: np.ndarray
    fragment_density: np.ndarray


@dataclass
class FragmentSputterCartesian:
    xs: np.ndarray
    ys: np.ndarray
    fragment_density: np.ndarray


@dataclass
class VectorialModelResult:
    volume_density_grid: np.ndarray
    volume_density: np.ndarray

    fragment_sputter: FragmentSputterSpherical

    collision_sphere_radius: Quantity
    max_grid_radius: Quantity
    coma_radius: Quantity

    column_density_grid: Optional[np.ndarray] = None
    column_density: Optional[np.ndarray] = None

    volume_density_interpolation: Optional[PPoly] = None
    column_density_interpolation: Optional[PPoly] = None

    coma: Optional[sba.VectorialModel] = None


def fragment_sputter_to_cartesian(
    fsp: Union[FragmentSputterSpherical, FragmentSputterPolar]
) -> FragmentSputterCartesian:
    """
    Fragment sputter information is generated in terms of (r, theta, density), so this converts to (x, y, density)
    with the appropriate type information so functions that care about sputter will know which coordinates are being provided
    """
    return FragmentSputterCartesian(
        xs=fsp.rs * np.sin(fsp.thetas),
        ys=fsp.rs * np.cos(fsp.thetas),
        fragment_density=fsp.fragment_density,
    )


def fragment_sputter_to_polar(fsp: FragmentSputterSpherical) -> FragmentSputterPolar:
    """Utility function to change the typing of given fragment sputter"""
    return FragmentSputterPolar(
        rs=fsp.rs, thetas=fsp.thetas, fragment_density=fsp.fragment_density
    )


def mirror_fragment_sputter(
    fsp: Union[FragmentSputterPolar, FragmentSputterCartesian]
) -> Union[FragmentSputterPolar, FragmentSputterCartesian]:
    """
    The sputter in (r, theta, density) in spherical coordinate format occupies the positive x-axis only due to the azimuthal symmetry of the problem.
    When we convert that to other coordinate systems, we might want to include the negative x-axis by mirroring the fragment sputter around x = 0,
    which this function provides.
    The type given is the return type.
    """
    if isinstance(fsp, FragmentSputterPolar):
        fsp.rs = np.append(fsp.rs, fsp.rs)
        fsp.thetas = np.append(fsp.thetas, -1 * fsp.thetas)
        fsp.fragment_density = np.append(fsp.fragment_density, fsp.fragment_density)
    elif isinstance(fsp, FragmentSputterCartesian):
        fsp.xs = np.append(fsp.xs, -1 * fsp.xs)
        fsp.ys = np.append(fsp.ys, fsp.ys)
        fsp.fragment_density = np.append(fsp.fragment_density, fsp.fragment_density)

    return fsp
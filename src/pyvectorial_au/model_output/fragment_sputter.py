from dataclasses import dataclass
import numpy as np


"""
Dataclasses for the two-dimensional fragment sputter information obtained from the model
"""


@dataclass
class FragmentSputterSpherical:
    """
    This holds the direct output of the model's fragment sputter in terms of (r, theta, density)
    Theta here is the polar angle in a spherical coordinate system, and the density is azimuthally symmetric
    All of the results are then 0 <= theta <= pi, occupying a half-plane
    """

    rs: np.ndarray
    thetas: np.ndarray
    fragment_density: np.ndarray


@dataclass
class FragmentSputterPolar:
    """
    If we want to expand the fragment density to the whole plane, we use this class to
    remove ambiguity of the domain of theta: 0 <= theta <= 2*pi
    We can then call mirror_fragment_sputter() to fill in the fragment_density to include
    the missing pi <= theta <= 2*pi section of the fragment sputter to have a full-plane
    representation
    """

    rs: np.ndarray
    thetas: np.ndarray
    fragment_density: np.ndarray


@dataclass
class FragmentSputterCartesian:
    """
    Representation of the fragment sputter in (x, y, density).  If we take results from
    the FragmentSputterSpherical, then only the positive x-axis half of the plane is included.
    Like FragmentSputterPolar, we can call mirror_fragment_sputter() to fill in the negative x-axis
    to obtain a whole plane view of the fragment density.
    """

    xs: np.ndarray
    ys: np.ndarray
    fragment_density: np.ndarray

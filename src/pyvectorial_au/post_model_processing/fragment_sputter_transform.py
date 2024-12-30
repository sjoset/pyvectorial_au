from typing import Union
import numpy as np

from pyvectorial_au.model_output.fragment_sputter import (
    FragmentSputterSpherical,
    FragmentSputterCartesian,
    FragmentSputterPolar,
)


def fragment_sputter_to_cartesian(
    fsp: Union[FragmentSputterSpherical, FragmentSputterPolar]
) -> FragmentSputterCartesian:
    """
    Fragment sputter information is generated by the model in terms of (r, theta, density), so this converts to (x, y, density)
    where theta is a spherical polar angle
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
    fsp: FragmentSputterPolar | FragmentSputterCartesian,
) -> FragmentSputterPolar | FragmentSputterCartesian:
    """
    The sputter in (r, theta, density) in spherical coordinate format occupies the positive x-axis only due to the azimuthal symmetry of the problem.
    When we convert that to other coordinate systems, we might want to include the negative x-axis by mirroring the fragment sputter around x = 0,
    which this function provides.
    The return type is the same type as the input
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

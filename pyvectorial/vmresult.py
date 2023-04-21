import copy
import numpy as np
import sbpy.activity as sba

from dataclasses import dataclass
from scipy.interpolate import PPoly
from astropy.units.quantity import Quantity

"""
    Defines VectorialModelResult, a dataclass for holding the result of either sbpy's VectorialModel or
    Festou's fortran version.  Functions written to handle VectorialModelResult don't need to care which model
    generated the results.

    Defines helper function to pull results out of sbpy model and stuff them into a VectorialModelResult.
    Functionality for taking a Fortran model result and producing a VectorialModelResult is provided in
    fortran_version.py
"""


# Dataclasses for the two-dimensional fragment sputter information obtained from the model
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
    column_density_grid: np.ndarray
    column_density: np.ndarray

    fragment_sputter: FragmentSputterPolar

    volume_density_interpolation: PPoly = None
    column_density_interpolation: PPoly = None

    collision_sphere_radius: Quantity = None
    max_grid_radius: Quantity = None
    coma_radius: Quantity = None

    num_fragments_theory: float = None
    num_fragments_grid: float = None


# Take a model run with sbpy and construct results
def get_result_from_coma(coma_orig: sba.VectorialModel) -> VectorialModelResult:
    coma = copy.deepcopy(coma_orig)

    vdg = coma.vmr.volume_density_grid
    vd = coma.vmr.volume_density
    cdg = coma.vmr.column_density_grid
    cd = coma.vmr.column_density
    vdi = coma.vmr.volume_density_interpolation
    cdi = coma.vmr.column_density_interpolation

    fs = FragmentSputterPolar(
        rs=coma.vmr.fragment_sputter.rs,
        thetas=coma.vmr.fragment_sputter.thetas,
        fragment_density=coma.vmr.fragment_sputter.fragment_density,
    )

    nft = coma.vmr.num_fragments_theory
    nfg = coma.vmr.num_fragments_grid

    csr = coma.vmr.collision_sphere_radius
    mgr = coma.vmr.max_grid_radius
    cr = coma.vmr.coma_radius

    return VectorialModelResult(
        volume_density_grid=vdg,
        volume_density=vd,
        column_density_grid=cdg,
        column_density=cd,
        fragment_sputter=fs,
        volume_density_interpolation=vdi,
        column_density_interpolation=cdi,
        collision_sphere_radius=csr,
        max_grid_radius=mgr,
        coma_radius=cr,
        num_fragments_theory=nft,
        num_fragments_grid=nfg,
    )


def cartesian_sputter_from_polar(fsp: FragmentSputterPolar) -> FragmentSputterCartesian:
    return FragmentSputterCartesian(
        xs=fsp.rs * np.sin(fsp.thetas),
        ys=fsp.rs * np.cos(fsp.thetas),
        fragment_density=fsp.fragment_density,
    )


def mirror_sputter(sp):
    if isinstance(sp, FragmentSputterPolar):
        sp.rs = np.append(sp.rs, sp.rs)
        sp.thetas = np.append(sp.thetas, -1 * sp.thetas)
        sp.fragment_density = np.append(sp.fragment_density, sp.fragment_density)
    elif isinstance(sp, FragmentSputterCartesian):
        sp.xs = np.append(sp.xs, -1 * sp.xs)
        sp.ys = np.append(sp.ys, sp.ys)
        sp.fragment_density = np.append(sp.fragment_density, sp.fragment_density)

    return sp

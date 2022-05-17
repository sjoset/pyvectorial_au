
import copy
import numpy as np
import sbpy.activity as sba

from dataclasses import dataclass
from scipy.interpolate import PPoly
from astropy.units.quantity import Quantity


@dataclass
class VectorialModelResult:
    volume_density_grid: np.ndarray
    volume_density: np.ndarray
    column_density_grid: np.ndarray
    column_density: np.ndarray
    angular_grid: np.ndarray
    volume_density_interpolation: PPoly
    column_density_interpolation: PPoly

    fragment_sputter: np.ndarray

    num_fragments_theory: float
    num_fragments_grid: float

    collision_sphere_radius: Quantity
    max_grid_radius: Quantity


def get_result_from_coma(coma_orig: sba.VectorialModel) -> VectorialModelResult:
    
    coma = copy.deepcopy(coma_orig)
    
    vdg = coma.vmodel['radial_grid']
    vd = coma.vmodel['radial_density']
    cdg = coma.vmodel['column_density_grid']
    cd = coma.vmodel['column_densities']
    vdi = coma.vmodel['r_dens_interpolation']
    cdi = coma.vmodel['column_density_interpolation']

    ag = coma.vmodel['angular_grid']
    fs = coma.vmodel['density_grid']

    nft = coma.vmodel['num_fragments_theory']
    nfg = coma.vmodel['num_fragments_grid']

    csr = coma.vmodel['collision_sphere_radius']
    mgr = coma.vmodel['max_grid_radius']

    return VectorialModelResult(
            volume_density_grid=vdg, volume_density=vd,
            column_density_grid=cdg, column_density=cd,
            volume_density_interpolation=vdi, column_density_interpolation=cdi,
            angular_grid=ag, fragment_sputter=fs,
            num_fragments_theory=nft, num_fragments_grid=nfg,
            collision_sphere_radius=csr, max_grid_radius=mgr
            )

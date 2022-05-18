
import copy
import numpy as np
import sbpy.activity as sba
import astropy.units as u

from dataclasses import dataclass
from scipy.interpolate import PPoly
from astropy.units.quantity import Quantity


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


def get_result_from_coma(coma_orig: sba.VectorialModel) -> VectorialModelResult:
    
    coma = copy.deepcopy(coma_orig)
    
    vdg = coma.vmodel['radial_grid']
    vd = coma.vmodel['radial_density']
    cdg = coma.vmodel['column_density_grid']
    cd = coma.vmodel['column_densities']
    vdi = coma.vmodel['r_dens_interpolation']
    cdi = coma.vmodel['column_density_interpolation']

    ag = coma.vmodel['angular_grid']
    fs_raw = coma.vmodel['density_grid']
    
    sputterlist = []
    for (i, j), frag_dens in np.ndenumerate(fs_raw):
        sputterlist.append([vdg[i].to(u.m).value, ag[j], frag_dens])
    sputter = np.array(sputterlist)
    rs = sputter[:, 0]
    thetas = sputter[:, 1]
    fragment_density = sputter[:, 2]

    fs = FragmentSputterPolar(rs=rs*u.m, thetas=thetas, fragment_density=fragment_density/u.m**3)

    nft = coma.vmodel['num_fragments_theory']
    nfg = coma.vmodel['num_fragments_grid']

    csr = coma.vmodel['collision_sphere_radius']
    mgr = coma.vmodel['max_grid_radius']
    cr  = coma.vmodel['coma_radius']

    return VectorialModelResult(
            volume_density_grid=vdg, volume_density=vd,
            column_density_grid=cdg, column_density=cd,
            fragment_sputter=fs,
            volume_density_interpolation=vdi, column_density_interpolation=cdi,
            collision_sphere_radius=csr, max_grid_radius=mgr, coma_radius=cr,
            num_fragments_theory=nft, num_fragments_grid=nfg
            )


def cartesian_sputter_from_polar(fsp: FragmentSputterPolar) -> FragmentSputterCartesian:

    return FragmentSputterCartesian(xs=fsp.rs*np.sin(fsp.thetas), ys=fsp.rs*np.cos(fsp.thetas), fragment_density=fsp.fragment_density)


def mirror_sputter(sp: FragmentSputterPolar | FragmentSputterCartesian) -> FragmentSputterPolar | FragmentSputterCartesian:

    if isinstance(sp, FragmentSputterPolar):
        sp.rs = np.append(sp.rs, sp.rs)
        sp.thetas = np.append(sp.thetas, -1*sp.thetas)
        sp.fragment_density = np.append(sp.fragment_density, sp.fragment_density)
    elif isinstance(sp, FragmentSputterCartesian):
        sp.xs = np.append(sp.xs, -1*sp.xs)
        sp.ys = np.append(sp.ys, sp.ys)
        sp.fragment_density = np.append(sp.fragment_density, sp.fragment_density)

    return sp


import numpy as np
import sbpy.activity as sba
import astropy.units as u
from dataclasses import dataclass

from .vmresult import VectorialModelResult

# [[r, theta, dens], [r, theta, dens]]

@dataclass
class FragmentSputterPolar:
    rs: np.ndarray
    thetas: np.ndarray
    fragment_density: np.ndarray

    r_limit: u.quantity.Quantity


@dataclass
class FragmentSputterCartesian:
    xs: np.ndarray
    ys: np.ndarray
    fragment_density: np.ndarray

    r_limit: u.quantity.Quantity


def fragment_sputter_from_fortran(fortran_sputter: np.ndarray, within_r: u.quantity.Quantity, mirrored=False) -> FragmentSputterCartesian:

    # fortran's radial sputter distances are output in km
    within_r_km = within_r.to(u.km).value

    # filter distances larger than within_r_km - fortran distances are in km
    sputter = np.array([x for x in fortran_sputter if x[0] < within_r_km])

    rs = sputter[:, 0]
    thetas = sputter[:, 1]
    fragment_density = sputter[:, 2]

    xs = rs*np.sin(thetas)
    ys = rs*np.cos(thetas)

    if mirrored:
        xs = np.append(xs, -1*xs)
        ys = np.append(ys, ys)
        fragment_density = np.append(fragment_density, fragment_density)

    # fortran's fragment sputter density is in 1/cm**3
    return FragmentSputterCartesian(xs=xs*u.km, ys=ys*u.km, fragment_density=fragment_density/u.cm**3, r_limit=within_r)


def fragment_sputter_from_sbpy(vmr: VectorialModelResult, within_r: u.quantity.Quantity, mirrored=False) -> FragmentSputterCartesian:

    # sbpy model works in meters
    within_r_m = within_r.to(u.m).value

    model_rs = vmr.volume_density_grid
    model_thetas = vmr.angular_grid

    # build array from the model results
    sputterlist = []
    for (i, j), frag_dens in np.ndenumerate(vmr.fragment_sputter):
        sputterlist.append([model_rs[i].to(u.m).value, model_thetas[j], frag_dens])
    sputter = np.array(sputterlist)

    # filter distances larger than within_r_m - sbpy distances are in m
    sputter = np.array([x for x in sputter if x[0] < within_r_m])

    rs = sputter[:, 0]
    thetas = sputter[:, 1]
    fragment_density = sputter[:, 2]

    xs = rs*np.sin(thetas)
    ys = rs*np.cos(thetas)

    if mirrored:
        xs = np.append(xs, -1*xs)
        ys = np.append(ys, ys)
        fragment_density = np.append(fragment_density, fragment_density)

    return FragmentSputterCartesian(xs=xs*u.m, ys=ys*u.m, fragment_density=fragment_density/u.m**3, r_limit=within_r)

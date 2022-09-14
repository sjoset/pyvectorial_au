
import numpy as np
import astropy.units as u
# import logging as log
import scipy.interpolate

from dataclasses import dataclass
from scipy import special, optimize
from typing import List, Callable
from functools import partial

from .haser_params import HaserParams


@dataclass
class HaserFitResult:
	fitting_function: None = None
	fitted_params: List = None
	covariances: List = None


@dataclass
class HaserScaleLengthSearchResult:
    # regular arrays
    parent_gammas: np.array = None
    fragment_gammas: np.array = None
    fitted_qs: np.array = None

    # meshgrids
    p_mesh: np.array = None
    f_mesh: np.array = None
    q_mesh: np.array = None

    # measure of agreement on total production
    # 0 is best: 'distance' of fitted production away from vectorial production
    agreements: np.array = None
    a_mesh: np.array = None

    # best fits for this search
    best_params: HaserParams = None


def _haser_column_density(rho_m, q_s, v_ms, p_m, f_m) -> Callable:

	"""
		Calculates the Haser column density at rho with given parameters to model,
		no astropy units attached, all quantities in meters and seconds
	"""
	sigma = q_s / (v_ms * rho_m * 2 * np.pi)
	sigma *= (f_m/(p_m - f_m)) * (special.iti0k0(rho_m/f_m)[1] - special.iti0k0(rho_m/p_m)[1])
	return sigma


def _make_haser_column_density_q(hps: HaserParams) -> Callable:

	"""
		Takes HaserParams and returns a column density function with Q as a fitting parameter
		
		Fitting function takes impact parameter in meters and returns column density in 1/m**2
	"""

	p_m = hps.gamma_p.to_value('m')
	f_m = hps.gamma_d.to_value('m')
	v_ms = hps.v_outflow.to_value('m/s')

	f = partial(_haser_column_density, v_ms=v_ms, p_m=p_m, f_m=f_m)
	return f


def haser_q_fit(q_guess: u.Quantity, hps: HaserParams, rs: np.ndarray, cds: np.ndarray) -> HaserFitResult:

    if hps.q is not None:
        print("Warning: do_haser_q_fit received non-empty production in HaserParams")

    hcd = _make_haser_column_density_q(hps)
    popt, pcov = optimize.curve_fit(hcd, rs, cds, p0=[q_guess.to_value('1/s')])
    return HaserFitResult(fitting_function=hcd, fitted_params=popt, covariances=pcov)


# TODO: test this
def haser_full_fit(q_guess, v_guess, parent_guess, fragment_guess, rs, cds):

    qg = q_guess.to_value('1/s')
    vg = v_guess.to_value('m/s')
    pg = parent_guess.to_value('m')
    fg = fragment_guess.to_value('m')

    hcd = _haser_column_density
    popt, pcov = optimize.curve_fit(hcd, rs, cds, p0=[qg, vg, pg, fg])
    return HaserFitResult(fitting_function=hcd, fitted_params=popt, covariances=pcov)


def find_best_haser_scale_lengths_q(vmc, vmr, parent_gammas: np.array, fragment_gammas: np.array) -> HaserScaleLengthSearchResult:

    # takes a finished vectorial model, fits Haser models of various scale lengths,
    # and looks for the (parent, fragment) scale length pair that agrees with the vectorial model's
    # input production

    num_parent_gammas = np.size(parent_gammas)
    num_fragment_gammas = np.size(fragment_gammas)

    fitting_results = []
    for parent_gamma in parent_gammas:
        for fragment_gamma in fragment_gammas:
            hps = HaserParams(q=None, v_outflow=vmc.parent.v_outflow, gamma_p=parent_gamma, gamma_d=fragment_gamma)
            hsr = haser_q_fit(q_guess=vmc.production.base_q, hps=hps, rs=vmr.column_density_grid, cds=vmr.column_density)
            # output is a list of the form [[parent, fragment, q_fitted], ...]
            fitting_results.append([parent_gamma.to_value('km'), fragment_gamma.to_value('km'), hsr.fitted_params[0]])

    fdata = np.array(fitting_results).reshape((num_parent_gammas*num_fragment_gammas, 3))

    result = HaserScaleLengthSearchResult

    # take columns of each variable
    result.parent_gammas = fdata[:, 0]
    result.fragment_gammas = fdata[:, 1]
    result.fitted_qs = fdata[:, 2]

    # generate meshgrids for plotting etc.
    result.p_mesh, result.f_mesh = np.meshgrid(np.unique(result.parent_gammas), np.unique(result.fragment_gammas))
    q_rbf = scipy.interpolate.Rbf(result.parent_gammas, result.fragment_gammas, result.fitted_qs, function='cubic')
    result.q_mesh = q_rbf(result.p_mesh, result.f_mesh)
    
    # find how much haser production agrees with given vectorial production
    result.agreements = np.sqrt((result.fitted_qs/vmc.production.base_q.to_value('1/s') - 1)**2)
    a_rbf = scipy.interpolate.Rbf(result.parent_gammas, result.fragment_gammas, result.agreements, function='cubic')
    result.a_mesh = a_rbf(result.p_mesh, result.f_mesh)

    # find index of minimum difference (best fit of productions)
    best = np.unravel_index(np.argmin(result.a_mesh, axis=None), result.a_mesh.shape)
    result.best_params = HaserParams(q=result.q_mesh[best], v_outflow=vmc.parent.v_outflow, gamma_p=result.p_mesh[best], gamma_d=result.f_mesh[best])

    return result

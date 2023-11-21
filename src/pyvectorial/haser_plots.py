import numpy as np

# import astropy.units as u
import matplotlib.pyplot as plt
import sbpy.activity as sba

from matplotlib import cm
from .haser_params import HaserParams
from .haser_fits import HaserScaleLengthSearchResult

# from .haser_fits import HaserScaleLengthSearchResult, haser_q_fit


"""
    Utility functions for Haser-related plotting
"""


def plot_haser_column_density(hps: HaserParams, ax: plt.Axes, rs: np.ndarray) -> None:
    """Take HaserParams and plot the column density along the values in the array rs"""

    coma = sba.Haser(Q=hps.q, v=hps.v_outflow, parent=hps.gamma_p, daughter=hps.gamma_d)
    ax.plot(rs, coma.column_density(rs))


def haser_search_result_plot(
    hsr: HaserScaleLengthSearchResult, ax: plt.Axes, colormap=cm.viridis
) -> None:
    """Take HaserScaleLengthSearchResult and make a 3d plot with contour projections of the results"""

    # contour plots of agreement on the 'floor' of the graph with a star to mark the best agreement
    ax.contour(
        hsr.p_mesh,
        hsr.f_mesh,
        hsr.a_mesh,
        zdir="z",
        offset=np.min(hsr.q_mesh) / 2,
        cmap=colormap,
        alpha=0.3,
    )
    ax.contourf(
        hsr.p_mesh,
        hsr.f_mesh,
        hsr.a_mesh,
        zdir="z",
        offset=np.min(hsr.q_mesh) / 2,
        cmap=colormap,
        alpha=0.3,
    )
    ax.scatter(
        hsr.best_params.gamma_p,
        hsr.best_params.gamma_d,
        np.min(hsr.q_mesh) / 2,
        marker="*",
    )

    # 3d surface of productions at each mesh point
    ax.plot_surface(
        hsr.p_mesh,
        hsr.f_mesh,
        hsr.q_mesh,
        rstride=1,
        cstride=1,
        cmap=colormap,
        alpha=0.8,
    )

    # flip x axis
    ax.set_xlim(ax.get_xlim()[::-1])


# def plot_fitted_haser_example(vmc, vmr):
#
#     hps = HaserParams(q=None, v_outflow=vmc.parent.v_outflow, gamma_p=sba.photo_lengthscale('H2O'), gamma_d=sba.photo_lengthscale('OH'))
#     hsr = haser_q_fit(q_guess=1.e28 * 1/u.s, hps=hps, rs=vmr.column_density_grid, cds=vmr.column_density)
#
#     plt.loglog(vmr.column_density_grid, vmr.column_density)
#     plt.plot(vmr.column_density_grid, hsr.fitting_function(vmr.column_density_grid.to_value('m'), *hsr.fitted_params))
#     plt.show()

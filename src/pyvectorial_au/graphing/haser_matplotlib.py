import numpy as np
import matplotlib.pyplot as plt
import sbpy.activity as sba

from pyvectorial.haser.haser_params import HaserParams

"""
    Utility functions for Haser-related plotting
"""


def plot_haser_column_density(hps: HaserParams, ax: plt.Axes, rs: np.ndarray) -> None:  # type: ignore
    """Take HaserParams and plot the column density along the values in the array rs"""

    coma = sba.Haser(Q=hps.q, v=hps.v_outflow, parent=hps.gamma_p, daughter=hps.gamma_d)
    ax.plot(rs, coma.column_density(rs))

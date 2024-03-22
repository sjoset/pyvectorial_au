import numpy as np
import astropy.units as u

from dataclasses import dataclass
from typing import Optional
from pyvectorial.model_input.vectorial_model_config import VectorialModelConfig


"""
    Like VectorialModelConfig, HaserParams serves as the standard input data structure, but for Haser models
"""


@dataclass
class HaserParams:
    # total parent production, ~ 1/u.s
    q: Optional[u.Quantity]
    # outflow speed of parents
    v_outflow: Optional[u.Quantity]
    # length scale of parent
    gamma_p: Optional[u.Quantity]
    # length scale of daughter
    gamma_d: Optional[u.Quantity]


def haser_from_vectorial_cd1980(vmc: VectorialModelConfig) -> HaserParams:
    # use relations in Combi & Delsemme 1980 to translate vectorial parameters
    # into roughly equivalent haser parameters

    v_d = np.sqrt(vmc.parent.v_outflow**2 + vmc.fragment.v_photo**2)

    gamma_p = vmc.parent.v_outflow * vmc.parent.tau_T
    gamma_d = v_d * vmc.fragment.tau_T

    delta = np.arctan(vmc.parent.v_outflow / vmc.fragment.v_photo)

    mu = gamma_p / gamma_d
    mu_h = mu * (mu + np.sin(delta)) / (1 + mu * np.sin(delta))

    gamma_d_h = np.sqrt((gamma_d**2 - gamma_p**2) / (mu_h**2 + 1))
    gamma_p_h = mu_h * gamma_d_h

    v_d_h = (v_d * gamma_d_h) / gamma_d

    # TODO: does the paper give a formula for the production, or should we use the same production as went
    # into the vectorial model?  If we fit this new haser and find the best Q, does it match the vectorial?
    # We return None for the production because ... ?
    return HaserParams(q=None, v_outflow=v_d_h, gamma_p=gamma_p_h, gamma_d=gamma_d_h)

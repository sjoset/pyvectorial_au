import logging as log
from typing import Callable, Optional

import astropy.units as u
import numpy as np

from pyvectorial_au.model_input.vectorial_model_config import (
    GaussianProductionTimeVariation,
    SineWaveProductionTimeVariation,
    SquarePulseProductionTimeVariation,
    VectorialModelConfig,
)


def make_time_dependence_function(vmc: VectorialModelConfig) -> Optional[Callable]:
    if isinstance(vmc.production.time_variation, GaussianProductionTimeVariation):
        return make_gaussian_q_t(vmc=vmc)
    elif isinstance(vmc.production.time_variation, SineWaveProductionTimeVariation):
        return make_sine_q_t(vmc=vmc)
    elif isinstance(vmc.production.time_variation, SquarePulseProductionTimeVariation):
        return make_square_pulse_q_t(vmc=vmc)
    else:
        return None


def make_gaussian_q_t(vmc: VectorialModelConfig) -> Callable:
    assert isinstance(vmc.production.time_variation, GaussianProductionTimeVariation)

    """Assembles a gaussian time dependence based on the parameters"""
    amplitude_in_invsecs = vmc.production.time_variation.amplitude.to_value(1 / u.s)  # type: ignore
    std_dev_in_secs = vmc.production.time_variation.std_dev.to_value(u.s)  # type: ignore
    t_max_in_secs = vmc.production.time_variation.t_max.to_value(u.s)  # type: ignore

    log.debug(
        "Building gaussian q_t:\tAmplitude: %s, t_max: %s, std_dev: %s",
        vmc.production.time_variation.amplitude,
        vmc.production.time_variation.t_max,
        vmc.production.time_variation.std_dev,
    )

    def q_t(t):
        return amplitude_in_invsecs * np.e ** -(
            ((t - t_max_in_secs) ** 2) / (2 * std_dev_in_secs**2)  # type: ignore
        )

    return q_t


def make_sine_q_t(vmc: VectorialModelConfig) -> Callable:
    assert isinstance(vmc.production.time_variation, SineWaveProductionTimeVariation)

    """Assembles a sinusoidal time dependence based on the parameters"""
    amplitude_in_invsecs = vmc.production.time_variation.amplitude.to_value(1 / u.s)  # type: ignore
    period_in_secs = vmc.production.time_variation.period.to_value(u.s)  # type: ignore
    delta_in_secs = vmc.production.time_variation.delta.to_value(u.s)  # type: ignore
    const_B = (2.0 * np.pi) / period_in_secs  # type: ignore

    log.debug(
        "Building sinusoidal q_t:\tAmplitude: %s, period: %s, delta: %s",
        vmc.production.time_variation.amplitude,
        vmc.production.time_variation.period,
        vmc.production.time_variation.delta,
    )

    def q_t(t):
        return amplitude_in_invsecs * (np.sin(const_B * (t + delta_in_secs)))

    return q_t


def make_square_pulse_q_t(vmc: VectorialModelConfig) -> Callable:
    assert isinstance(vmc.production.time_variation, SquarePulseProductionTimeVariation)

    """Assembles a square pulse time dependence based on the parameters"""
    t_start_in_secs = vmc.production.time_variation.t_start.to_value(u.s)  # type: ignore
    tend_in_secs = (  # type: ignore
        vmc.production.time_variation.t_start - vmc.production.time_variation.duration
    ).to_value(  # type: ignore
        u.s
    )
    amplitude_in_invsecs = vmc.production.time_variation.amplitude.to_value(1 / u.s)  # type: ignore

    log.debug(
        "Building square pulse q_t:\tAmplitude: %s, t_start: %s, duration: %s",
        vmc.production.time_variation.amplitude,
        vmc.production.time_variation.t_start,
        vmc.production.time_variation.duration,
    )

    def q_t(t):
        # Comparisons seem backward because of our weird time system
        if t < t_start_in_secs and t > tend_in_secs:
            return amplitude_in_invsecs
        else:
            return 0.0

    return q_t

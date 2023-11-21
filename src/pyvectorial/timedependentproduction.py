from enum import StrEnum
import logging as log

import astropy.units as u
import numpy as np


class TimeDependentProductionType(StrEnum):
    square_pulse = "square pulse"
    gaussian = "gaussian"
    sine_wave = "sine wave"


class TimeDependentProduction:
    """
    Class for generating time-dependent production functions to hand to
    the vectorial model included in sbpy

    Create the class by passing it a supported type, and get the time function
    by calling create_production

    Currently supported: square pulses, gaussians, and sine waves
    """

    def __init__(self, production_type: TimeDependentProductionType):
        # self.supported_types = {"square pulse", "gaussian", "sine wave"}
        self.production_type = production_type

    def create_production(self, **kwargs):
        if self.production_type == TimeDependentProductionType.square_pulse:
            allowed_keys = {"amplitude", "t_start", "duration"}
            self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
            return self.make_square_pulse_q_t()
        elif self.production_type == TimeDependentProductionType.gaussian:
            allowed_keys = {"amplitude", "t_max", "std_dev"}
            self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
            return self.make_gaussian_q_t()
        elif self.production_type == TimeDependentProductionType.sine_wave:
            allowed_keys = {"amplitude", "period", "delta"}
            self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
            return self.make_sine_q_t()
        else:
            return None

    def make_square_pulse_q_t(self):
        """Assembles a square pulse time dependence based on the parameters"""
        t_start_in_secs = self.t_start.to(u.s).value  # type: ignore
        tend_in_secs = (self.t_start - self.duration).to(u.s).value  # type: ignore
        amplitude_in_invsecs = self.amplitude.to(1 / (u.s)).value  # type: ignore

        log.debug(
            "Building square pulse q_t:\tAmplitude: %s, t_start: %s, duration: %s",
            self.amplitude,  # type: ignore
            self.t_start,  # type: ignore
            self.duration,  # type: ignore
        )

        def q_t(t):
            # Comparisons seem backward because of our weird time system
            if t < t_start_in_secs and t > tend_in_secs:
                return amplitude_in_invsecs
            else:
                return 0.0

        return q_t

    def make_gaussian_q_t(self):
        """Assembles a gaussian time dependence based on the parameters"""
        t_max_in_secs = self.t_max.to(u.s).value  # type: ignore
        std_dev_in_secs = self.std_dev.to(u.s).value  # type: ignore
        amplitude_in_invsecs = self.amplitude.to(1 / (u.s)).value  # type: ignore

        log.debug(
            "Building gaussian q_t:\tAmplitude: %s, t_max: %s, std_dev: %s",
            self.amplitude,  # type: ignore
            self.t_max,  # type: ignore
            self.std_dev,  # type: ignore
        )

        def q_t(t):
            return amplitude_in_invsecs * np.e ** -(
                ((t - t_max_in_secs) ** 2) / (2 * std_dev_in_secs**2)
            )

        return q_t

    def make_sine_q_t(self):
        """Assembles a sinusoidal time dependence based on the parameters"""
        period_in_secs = self.period.to(u.s).value  # type: ignore
        delta_in_secs = self.delta.to(u.s).value  # type: ignore
        amplitude_in_invsecs = self.amplitude.to(1 / (u.s)).value  # type: ignore
        const_B = (2.0 * np.pi) / period_in_secs

        log.debug(
            "Building sinusoidal q_t:\tAmplitude: %s, period: %s, delta: %s",
            self.amplitude,  # type: ignore
            self.period,  # type: ignore
            self.delta,  # type: ignore
        )

        def q_t(t):
            return amplitude_in_invsecs * (np.sin(const_B * (t + delta_in_secs)))

        return q_t

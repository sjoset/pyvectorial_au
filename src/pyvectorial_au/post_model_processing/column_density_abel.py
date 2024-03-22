import logging as log
import numpy as np
import astropy.units as u

from abel.hansenlaw import hansenlaw_transform
from pyvectorial_au.model_output.vectorial_model_result import VectorialModelResult


def column_density_from_abel(vmr: VectorialModelResult, num_samples=10000) -> None:
    # we want to sample the volume density over the entire grid with a linspace:
    # the reverse Abel transform needs a constant delta between points
    sample_space_start = np.min(vmr.volume_density_grid)
    sample_space_end = np.max(vmr.volume_density_grid)
    sample_space, dr = np.linspace(
        start=sample_space_start.to_value(u.m),
        stop=sample_space_end.to_value(u.m),
        num=num_samples,
        retstep=True,
    )

    if vmr.volume_density_interpolation is None:
        log.debug(
            "Cannot add column density with Abel transform: no volume density interpolation available!"
        )
        return

    sampled_volume_densities = vmr.volume_density_interpolation(sample_space)

    abel_column_densities = hansenlaw_transform(
        sampled_volume_densities, dr=dr, direction="forward"
    )

    vmr.column_density_grid = sample_space * u.m
    vmr.column_density = abel_column_densities / u.m**2  # type: ignore

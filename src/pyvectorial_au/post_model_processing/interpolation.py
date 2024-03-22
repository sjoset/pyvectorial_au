import logging as log
import astropy.units as u

from scipy.interpolate import CubicSpline

from pyvectorial_au.model_output.vectorial_model_result import VectorialModelResult


def interpolate_volume_density(vmr: VectorialModelResult) -> None:
    if vmr.volume_density_interpolation is not None:
        log.debug(
            "Attempted to add volume density interpolation to a VectorialModelResult that already had one! Skipping."
        )
        return

    vmr.volume_density_interpolation = CubicSpline(
        vmr.volume_density_grid.to_value(u.m),  # type: ignore
        vmr.volume_density.to_value(1 / u.m**3),  # type: ignore
        bc_type="natural",
    )


def interpolate_column_density(vmr: VectorialModelResult) -> None:
    if vmr.column_density_interpolation is not None:
        log.debug(
            "Attempted to add column density interpolation to a VectorialModelResult that already had one! Skipping."
        )
        return

    if vmr.column_density_grid is None or vmr.column_density is None:
        log.debug(
            "Attempted to add column density interpolation to a VectorialModelResult that had no column density! Skipping."
        )
        return

    vmr.column_density_interpolation = CubicSpline(
        vmr.column_density_grid.to_value(u.m),  # type: ignore
        vmr.column_density.to_value(1 / u.m**2),  # type: ignore
        bc_type="natural",
    )

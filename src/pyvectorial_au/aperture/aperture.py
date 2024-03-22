import logging as log
from typing import Tuple

import numpy as np
import scipy.integrate
import astropy.units as u
import sbpy.activity as sba

from pyvectorial.model_output.vectorial_model_result import VectorialModelResult


class UncenteredRectangularAperture(sba.Aperture):
    """Rectangular aperture projected at the distance of the target.

    Parameters
    ----------
    shape : `~astropy.units.Quantity`
        A four-element `~astropy.units.Quantity` of angular or
        projected linear size for the width and height of the
        aperture.  The order is not significant.

    """

    def __init__(self, shape):
        if len(shape) != 4:
            raise ValueError("shape must be 4 elements")
        super().__init__(shape)

    def __repr__(self):
        return f"<UncenteredRectangularAperture: coordinates {self.dim[0].value},{self.dim[1].value}×{self.dim[2].value},{self.dim[3].value}>"

    @property
    def shape(self):
        """Rectangle dimensions."""
        return self.dim

    def coma_equivalent_radius(self):
        # this function only makes sense when the aperture is centered
        return 0

    coma_equivalent_radius.__doc__ = sba.Aperture.coma_equivalent_radius.__doc__


def total_number_in_aperture(
    vmr: VectorialModelResult, ap: sba.Aperture, epsabs=1.49e-8
) -> Tuple[float, float]:
    """
    Integrate column density over an aperture, returning tuple of (result, err)
    """

    if vmr.column_density_interpolation is None:
        log.debug(
            "Attempting total_number_in_aperture but column density interpolation is missing!"
        )
        return (0.0, 0.0)

    if isinstance(ap, (sba.CircularAperture, sba.AnnularAperture)):
        if isinstance(ap, sba.CircularAperture):
            limits = (0, ap.radius.to_value(u.m))
        else:
            limits = ap.shape.to_value(u.m)

        # integrate in polar coordinates
        def f_circular(rho):
            """Column density integration in polar coordinates.

            rho in m, column_density in m**-2

            """
            return rho * vmr.column_density_interpolation(rho)  # type: ignore

        N, err = scipy.integrate.quad(f_circular, *limits, epsabs=epsabs)
        N *= 2 * np.pi
        err *= 2 * np.pi
    elif isinstance(ap, sba.RectangularAperture):
        shape = ap.shape.to_value(u.m)

        def f_rectangular(rho, _):
            """Column density integration in polar coordinates.

            rho in m, column_density in m**-2

            th is ignored (azimuthal symmetry)

            """
            return rho * vmr.column_density_interpolation(rho)  # type: ignore

        # first "octant"; rho1 and rho2 are the limits of the
        # integration
        def rho1(_):
            "Lower limit"
            return 0

        def rho2_1(th):
            "Upper limit (a line)"
            return shape[0] / 2 / np.cos(th)

        th = np.arctan(shape[1] / shape[0])
        N1, err1 = scipy.integrate.dblquad(  # type: ignore
            f_rectangular, 0, th, rho1, rho2_1, epsabs=epsabs
        )

        # second "octant"
        def rho2_2(th):
            "Upper limit (a line)"
            return shape[1] / 2 / np.cos(th)

        th = np.arctan(shape[0] / shape[1])
        N2, err2 = scipy.integrate.dblquad(  # type: ignore
            f_rectangular, 0, th, rho1, rho2_2, epsabs=epsabs
        )

        # N1 + N2 constitute 1/4th of the rectangle
        N = 4 * (N1 + N2)
        err = 4 * (err1 + err2)
    elif isinstance(ap, sba.GaussianAperture):
        # integrate in polar coordinates
        def f_gauss(rho, sigma):
            """Column density integration in polar coordinates.

            rho and sigma in m, column_density in m**-2

            """
            return (
                rho
                * np.exp(-(rho**2) / sigma**2 / 2)
                * vmr.column_density_interpolation(rho)  # type: ignore
            )

        sigma = ap.sigma.to_value(u.m)
        N, err = scipy.integrate.quad(f_gauss, 0, np.inf, args=(sigma,), epsabs=epsabs)
        N *= 2 * np.pi
        err *= 2 * np.pi
    elif isinstance(ap, UncenteredRectangularAperture):
        shape = ap.shape.to_value(u.m)

        def f_uncentered(x, y):
            rho = np.sqrt(x**2 + y**2)
            return vmr.column_density_interpolation(rho)  # type: ignore

        # shape = (x1, y1, x2, y2)
        N, err = scipy.integrate.dblquad(  # type: ignore
            f_uncentered, shape[0], shape[2], shape[1], shape[3], epsabs=epsabs
        )
    else:
        N, err = 0, 0
        log.debug("Aperture type %s not handled in total_number_in_aperture!", type(ap))

    return N, err

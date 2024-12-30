import sbpy.activity as sba


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

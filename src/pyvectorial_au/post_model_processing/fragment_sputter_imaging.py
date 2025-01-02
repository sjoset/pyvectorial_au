import numpy as np
import astropy.units as u
from scipy.interpolate import griddata

from pyvectorial_au.model_output.fragment_sputter import (
    FragmentSputterPolar,
    FragmentSputterSpherical,
)
from pyvectorial_au.post_model_processing.fragment_sputter_transform import (
    fragment_sputter_to_polar,
    mirror_fragment_sputter,
)


def fragment_sputter_volume_density_to_image(
    fragment_sputter: FragmentSputterSpherical,
    view_within_radius: u.Quantity,
    image_size: int,
    dist_units: u.Quantity = u.km,  # type: ignore
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a image of the provided fragment sputter, along with the x and y mesh grid

    view_within_radius: visualize the fragment sputter around the comet for all r < view_within_radius
    image_size: pixel size of the generated image (width and length will be equal to image_size)
    dist_units: units for the spatial grid that corresponds to each pixel

    returns:
    (image, grid_x, grid_y)
    """

    fs: FragmentSputterPolar = fragment_sputter_to_polar(fragment_sputter)
    fs: FragmentSputterPolar = mirror_fragment_sputter(fs)  # type: ignore

    rs = fs.rs
    thetas = fs.thetas
    xs = rs * np.sin(thetas)
    ys = rs * np.cos(thetas)
    zs = fs.fragment_density.to_value(1 / dist_units**3)  # type: ignore

    x_max = view_within_radius.to_value(dist_units)
    x_min = -1 * x_max  # type: ignore
    y_max = view_within_radius.to_value(dist_units)
    y_min = -1 * y_max  # type: ignore

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, image_size), np.linspace(y_min, y_max, image_size)  # type: ignore
    )

    # interpolate pixel values onto our pixel grid
    fs_img = griddata(
        points=(xs, ys),
        values=zs,
        xi=(grid_x, grid_y),
        method="cubic",
        fill_value=0,
    )

    return fs_img, grid_x, grid_y

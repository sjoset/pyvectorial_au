import numpy as np
import copy
import astropy.units as u
import scipy.interpolate
import plotly.graph_objects as go
from typing import Tuple
from typing import Optional

from pyvectorial.model_output.vectorial_model_result import (
    VectorialModelResult,
    FragmentSputterPolar,
    FragmentSputterSpherical,
    fragment_sputter_to_cartesian,
    mirror_fragment_sputter,
)


"""
    Functions for using plotly to plot various data contained in VectorialModelResults
"""

# color palette
myred = "#c74a77"
mybred = "#dbafad"
mygreen = "#afac7c"
mybgreen = "#dbd89c"
mypeach = "#dbb89c"
mybpeach = "#e9d4c3"
myblue = "#688894"
mybblue = "#a4b7be"
myblack = "#301e2a"
mybblack = "#82787f"
mywhite = "#d8d7dc"
mybwhite = "#e7e7ea"

# TODO: port over the column density inflection point marker in the matplotlib version?  Collision sphere marker
# function as well?

_default_vdens_units = 1 / u.m**3  # type: ignore


def plotly_volume_density_plot(
    vmr: VectorialModelResult,
    dist_units=u.m,
    vdens_units=_default_vdens_units,
    **kwargs
) -> go.Scatter:
    assert isinstance(vmr.volume_density, u.Quantity)
    assert isinstance(vmr.volume_density_grid, u.Quantity)

    xs = vmr.volume_density_grid.to(dist_units)
    ys = vmr.volume_density.to(vdens_units)

    vdens_plot = go.Scatter(x=xs, y=ys, **kwargs)
    return vdens_plot


def plotly_volume_density_interpolation_plot(
    vmr: VectorialModelResult,
    dist_units=u.m,
    vdens_units=_default_vdens_units,
    **kwargs
) -> go.Scatter:
    # model's interpolation function needs meters in, gives output in 1/m**3
    ys = (
        vmr.volume_density_interpolation(vmr.volume_density_grid.to_value(u.m)) / u.m**3  # type: ignore
    )

    vdens_interp_plot = go.Scatter(
        x=vmr.volume_density_grid.to_value(dist_units),  # type: ignore
        y=ys.to_value(vdens_units),
        **kwargs
    )
    return vdens_interp_plot


def plotly_column_density_plot(
    vmr: VectorialModelResult, dist_units=u.m, cdens_units=1 / u.m**2, **kwargs  # type: ignore
) -> go.Scatter:
    xs = vmr.column_density_grid.to(dist_units)  # type: ignore
    ys = vmr.column_density.to(cdens_units)  # type: ignore

    cdens_plot = go.Scatter(x=xs, y=ys, **kwargs)
    return cdens_plot


def plotly_column_density_interpolation_plot(
    vmr: VectorialModelResult, dist_units=u.m, cdens_units=1 / u.m**2, **kwargs  # type: ignore
) -> go.Scatter:
    # model's interpolation function needs meters in, gives output in 1/m**2
    ys = (
        vmr.column_density_interpolation(vmr.column_density_grid.to_value(u.m)) / u.m**2  # type: ignore
    )

    cdens_interp_plot = go.Scatter(
        x=vmr.column_density_grid.to_value(dist_units),  # type: ignore
        y=ys.to_value(cdens_units),
        **kwargs
    )
    return cdens_interp_plot


def plotly_column_density_plot_3d(
    vmr: VectorialModelResult,
    center=(0, 0) * u.km,  # type: ignore
    width=200000 * u.km,  # type: ignore
    height=200000 * u.km,  # type: ignore
    divisions=100,
    dist_units=u.m,
    cdens_units=1 / u.m**2,  # type: ignore
    **kwargs
) -> go.Surface:
    xmin_m, ymin_m = np.subtract(
        center.to_value(u.m), (width.to_value(u.m) / 2, height.to_value(u.m) / 2)
    )
    xmax_m, ymax_m = np.add(
        (xmin_m, ymin_m), (width.to_value(u.m), height.to_value(u.m))
    )
    xs_m = np.linspace(xmin_m, xmax_m, num=divisions)
    ys_m = np.linspace(ymin_m, ymax_m, num=divisions)

    x_mesh_m, y_mesh_m = np.meshgrid(xs_m, ys_m)
    z_mesh = (
        vmr.column_density_interpolation(np.sqrt(x_mesh_m**2 + y_mesh_m**2)) / u.m**2  # type: ignore
    )
    x_mesh = x_mesh_m * u.m
    y_mesh = y_mesh_m * u.m

    cdens_plot = go.Surface(
        x=x_mesh.to_value(dist_units),
        y=y_mesh.to_value(dist_units),
        z=z_mesh.to_value(cdens_units),
        **kwargs
    )
    return cdens_plot


def plotly_fragment_sputter_contour_plot(
    vmr,
    dist_units=u.km,
    sputter_units=1 / u.cm**3,
    within_r=2000 * u.km,  # type: ignore
    min_r=0 * u.km,  # type: ignore
    max_angle=np.pi,
    mirrored=False,
    show_outflow_axis=True,
    **kwargs
) -> Tuple[go.Contour, Optional[go.Scatter], float]:
    fragment_sputter = copy.deepcopy(vmr.fragment_sputter)

    if mirrored:
        fragment_sputter = mirror_fragment_sputter(fragment_sputter)

    within_max_angle = fragment_sputter.thetas < max_angle  # type: ignore
    fragment_sputter.rs = fragment_sputter.rs[within_max_angle]  # type: ignore
    fragment_sputter.thetas = fragment_sputter.thetas[within_max_angle]  # type: ignore
    fragment_sputter.fragment_density = fragment_sputter.fragment_density[
        within_max_angle
    ]

    if isinstance(fragment_sputter, FragmentSputterPolar) or isinstance(
        fragment_sputter, FragmentSputterSpherical
    ):
        fragment_sputter = fragment_sputter_to_cartesian(fragment_sputter)

    xs = fragment_sputter.xs
    ys = fragment_sputter.ys
    zs = fragment_sputter.fragment_density

    within_limit = np.logical_and(
        np.sqrt(xs**2 + ys**2) < within_r, np.sqrt(xs**2 + ys**2) > min_r
    )
    xs = xs[within_limit].to_value(dist_units)  # type: ignore
    ys = ys[within_limit].to_value(dist_units)  # type: ignore
    zs = zs[within_limit].to_value(sputter_units)  # type: ignore

    x_mesh, y_mesh = np.meshgrid(np.unique(xs), np.unique(ys))
    fs_rbf = scipy.interpolate.Rbf(xs, ys, zs, function="cubic")
    frag_mesh = fs_rbf(x_mesh, y_mesh)

    sputter_contour = go.Contour(
        x=np.unique(xs), y=np.unique(ys), z=frag_mesh, **kwargs
    )

    if show_outflow_axis:
        # highlight the outflow axis, along positive y
        outflow_axis = go.Scatter(
            x=[0, 0], y=[0, np.max(ys) * 0.9], mode="lines", opacity=0.5
        )
    else:
        outflow_axis = None

    return (sputter_contour, outflow_axis, np.max(xs))


def plotly_fragment_sputter_plot(
    vmr,
    dist_units=u.m,
    sputter_units=1 / u.m**3,  # type: ignore
    within_r=5000 * u.km,  # type: ignore
    min_r=0 * u.km,  # type: ignore
    mirrored=False,
    show_outflow_axis=True,
    **kwargs
) -> Tuple[Optional[go.Scatter3d], Optional[go.Scatter3d], Optional[float]]:
    fragment_sputter = copy.deepcopy(vmr.fragment_sputter)

    if mirrored:
        fragment_sputter = mirror_fragment_sputter(fragment_sputter)

    if isinstance(fragment_sputter, FragmentSputterPolar) or isinstance(
        fragment_sputter, FragmentSputterSpherical
    ):
        fragment_sputter = fragment_sputter_to_cartesian(fragment_sputter)

    # if mirrored:
    #     fragment_sputter = mirror_sputter(fragment_sputter)
    #
    # if isinstance(fragment_sputter, FragmentSputterPolar):
    #     fragment_sputter = cartesian_sputter_from_polar(fragment_sputter)

    xs = fragment_sputter.xs
    ys = fragment_sputter.ys
    zs = fragment_sputter.fragment_density

    # above_limit = np.sqrt(xs**2 + ys**2) > min_r
    # xs = xs[above_limit]
    # ys = ys[above_limit]
    # zs = zs[above_limit]

    within_limit = np.logical_and(
        np.sqrt(xs**2 + ys**2) < within_r, np.sqrt(xs**2 + ys**2) > min_r
    )
    if not len(xs):
        print("Radial cutoff for fragment sputter too small!  Nothing to plot.")
        return (None, None, None)

    xs = xs[within_limit].to_value(dist_units)  # type: ignore
    ys = ys[within_limit].to_value(dist_units)  # type: ignore
    zs = zs[within_limit].to_value(sputter_units)  # type: ignore

    sputter_plot = go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers", marker_color=zs, **kwargs
    )

    # highlight the outflow axis, along positive y
    if show_outflow_axis:
        outflow_axis = go.Scatter3d(
            x=[0, 0], y=[0, np.max(ys) * 0.9], z=[0, 0], mode="lines"
        )
    else:
        outflow_axis = None

    return (sputter_plot, outflow_axis, np.max(xs))


def plotly_q_t_plot(
    vmr: VectorialModelResult, time_units=u.hour
) -> Optional[go.Scatter]:
    if vmr.coma is None:
        return None

    # coma q_t function takes seconds, no astropy units attached
    ts_s = (np.linspace(-40, 40, num=1000) * u.hour).to_value(u.s)  # type: ignore
    f_q = np.vectorize(vmr.coma.q_t)
    qs = f_q(ts_s)
    t_h = (ts_s * u.s).to_value(time_units)

    qtplot = go.Scatter(x=t_h, y=qs, mode="lines")
    return qtplot

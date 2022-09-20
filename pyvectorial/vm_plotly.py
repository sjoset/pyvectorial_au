#!/usr/bin/env python3

import numpy as np
import astropy.units as u
import scipy.interpolate
import plotly.graph_objects as go
from typing import Tuple
from sbpy.activity import VectorialModel

# import matplotlib.pyplot as plt
# import matplotlib.cm as cmx
# from matplotlib.colors import Normalize

from .vmresult import VectorialModelResult, FragmentSputterPolar, cartesian_sputter_from_polar, mirror_sputter

# solarbluecol = np.array([38, 139, 220]) / 255.
# solarblue = (solarbluecol[0], solarbluecol[1], solarbluecol[2], 1)
# solargreencol = np.array([133, 153, 0]) / 255.
# solargreen = (solargreencol[0], solargreencol[1], solargreencol[2], 1)
# solarblackcol = np.array([0, 43, 54]) / 255.
# solarblack = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)
# solarwhitecol = np.array([238, 232, 213]) / 255.
# solarwhite = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)

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


# def _find_cdens_inflection_points(vmr: VectorialModelResult) -> np.ndarray:
#     """
#         Look for changes in sign of second derivative of the column density,
#         given a VectorialModelResult and return a list of inflection points
#     """
#
#     xs = np.linspace(0, 5e8, num=100)
#     concavity = vmr.column_density_interpolation.derivative(nu=2)
#     ys = concavity(xs)
#
#     # for pair in zip(xs, ys):
#     #     print(f"R: {pair[0]:08.1e}\t\tConcavity: {pair[1]:8.8f}")
#
#     # Array of 1s or 0s marking if the sign changed from one element to the next
#     sign_changes = (np.diff(np.sign(ys)) != 0)*1
#
#     # Manually remove the weirdness near the nucleus and rezize the array
#     sign_changes[0] = 0
#     sign_changes = np.resize(sign_changes, 100)
#
#     inflection_points = xs*sign_changes
#     # Only want non-zero elements
#     inflection_points = inflection_points[inflection_points > 0]
#
#     # Only want inflection points outside the collision sphere
#     csphere_radius = vmr.collision_sphere_radius.to_value(u.m)
#     inflection_points = inflection_points[inflection_points > csphere_radius]
#
#     inflection_points = inflection_points * u.m
#     return inflection_points


# def plotly_mark_inflection_points(vmr: VectorialModelResult, ax, **kwargs) -> None:
#
#     # Find possible inflection points
#     for ipoint in _find_cdens_inflection_points(vmr):
#         ax.axvline(x=ipoint, **kwargs)


# def plotly_mark_collision_sphere(vmr: VectorialModelResult, ax, **kwargs) -> None:
#
#     # Mark the beginning of the collision sphere
#     ax.axvline(x=vmr.collision_sphere_radius, **kwargs)


def plotly_volume_density_plot(vmr: VectorialModelResult, dist_units=u.m, vdens_units=1/u.m**3, **kwargs) -> go.Scatter:

    xs = vmr.volume_density_grid.to(dist_units)
    ys = vmr.volume_density.to(vdens_units)
    
    vdens_plot = go.Scatter(x=xs, y=ys, **kwargs)
    return vdens_plot


def plotly_volume_density_interpolation_plot(vmr: VectorialModelResult, dist_units=u.m, vdens_units=1/u.m**3, **kwargs) -> go.Scatter:

    # model's interpolation function needs meters in, gives output in 1/m**3
    ys = vmr.volume_density_interpolation(vmr.volume_density_grid.to_value(u.m))/u.m**3

    vdens_interp_plot = go.Scatter(x=vmr.volume_density_grid.to_value(dist_units), y=ys.to_value(vdens_units), **kwargs)
    return vdens_interp_plot


def plotly_column_density_plot(vmr: VectorialModelResult, dist_units=u.m, cdens_units=1/u.m**2, **kwargs) -> go.Scatter:

    xs = vmr.column_density_grid.to(dist_units)
    ys = vmr.column_density.to(cdens_units)

    cdens_plot = go.Scatter(x=xs, y=ys, **kwargs)
    return cdens_plot


def plotly_column_density_interpolation_plot(vmr: VectorialModelResult, dist_units=u.m, cdens_units=1/u.m**2, **kwargs) -> go.Scatter:

    # model's interpolation function needs meters in, gives output in 1/m**2
    ys = vmr.column_density_interpolation(vmr.column_density_grid.to_value(u.m))/u.m**2

    cdens_interp_plot = go.Scatter(x=vmr.column_density_grid.to_value(dist_units), y=ys.to_value(cdens_units), **kwargs)
    return cdens_interp_plot


def plotly_column_density_plot_3d(vmr: VectorialModelResult, center=(0,0)*u.km, width=200000*u.km, height=200000*u.km, divisions=100, dist_units=u.m, cdens_units=1/u.m**2, **kwargs) -> go.Surface:

    xmin_m, ymin_m = np.subtract(center.to_value(u.m), (width.to_value(u.m)/2, height.to_value(u.m)/2))
    xmax_m, ymax_m = np.add((xmin_m, ymin_m), (width.to_value(u.m), height.to_value(u.m)))
    xs_m = np.linspace(xmin_m, xmax_m, num=divisions)
    ys_m = np.linspace(ymin_m, ymax_m, num=divisions)

    x_mesh_m, y_mesh_m = np.meshgrid(xs_m, ys_m)
    z_mesh = vmr.column_density_interpolation(np.sqrt(x_mesh_m**2 + y_mesh_m**2))/u.m**2
    x_mesh = x_mesh_m * u.m
    y_mesh = y_mesh_m * u.m

    cdens_plot = go.Surface(x=x_mesh.to_value(dist_units), y=y_mesh.to_value(dist_units), z=z_mesh.to_value(cdens_units), **kwargs)
    return cdens_plot


def plotly_fragment_sputter_contour_plot(vmr, dist_units=u.km, sputter_units=1/u.cm**3, within_r=10000*u.km, mirrored=False, show_outflow_axis=True, **kwargs) -> Tuple[go.Contour, go.Scatter, float]:

    fsc = vmr.fragment_sputter

    if mirrored:
        fsc = mirror_sputter(fsc)

    if isinstance(fsc, FragmentSputterPolar):
        fsc = cartesian_sputter_from_polar(fsc)

    xs = fsc.xs
    ys = fsc.ys
    zs = fsc.fragment_density

    within_limit = np.sqrt(xs**2 + ys**2) < within_r
    xs = xs[within_limit].to_value(dist_units)
    ys = ys[within_limit].to_value(dist_units)
    zs = zs[within_limit].to_value(sputter_units)

    x_mesh, y_mesh = np.meshgrid(np.unique(xs), np.unique(ys))
    fs_rbf = scipy.interpolate.Rbf(xs, ys, zs, function='cubic')
    frag_mesh = fs_rbf(x_mesh, y_mesh)

    sputter_contour = go.Contour(x=np.unique(xs), y=np.unique(ys), z=frag_mesh, **kwargs)

    if show_outflow_axis:
        # highlight the outflow axis, along positive y
        outflow_axis = go.Scatter(x=[0, 0], y=[0, np.max(ys)*0.9], mode='lines', opacity=0.5)
    else:
        outflow_axis = None

    return (sputter_contour, outflow_axis, np.max(xs))


def plotly_fragment_sputter_plot(vmr, dist_units=u.m, sputter_units=1/u.m**3, within_r=1000*u.km, mirrored=False, show_outflow_axis=True, **kwargs) -> Tuple[go.Contour, go.Scatter, float]:

    fsc = vmr.fragment_sputter

    if mirrored:
        fsc = mirror_sputter(fsc)

    if isinstance(fsc, FragmentSputterPolar):
        fsc = cartesian_sputter_from_polar(fsc)

    xs = fsc.xs
    ys = fsc.ys
    zs = fsc.fragment_density

    within_limit = np.sqrt(xs**2 + ys**2) < within_r
    xs = xs[within_limit].to_value(dist_units)
    ys = ys[within_limit].to_value(dist_units)
    zs = zs[within_limit].to_value(sputter_units)

    sputter_plot = go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker_color=zs, **kwargs)

    # highlight the outflow axis, along positive y
    if show_outflow_axis:
        outflow_axis = go.Scatter3d(x=[0, 0], y=[0, np.max(ys)*0.9], z=[0, 0], mode='lines')
    else:
        outflow_axis = None

    return (sputter_plot, outflow_axis, np.max(xs))


def plotly_q_t_plot(coma: VectorialModel, time_units=u.hour) -> go.Scatter:

    # coma q_t function takes seconds, no astropy units attached
    ts_s = (np.linspace(-40, 40, num=1000) * u.hour).to_value(u.s)
    f_q = np.vectorize(coma.q_t)
    qs = f_q(ts_s)
    t_h = (ts_s * u.s).to_value(time_units)

    qtplot = go.Scatter(x=t_h, y=qs, mode='lines')
    return qtplot

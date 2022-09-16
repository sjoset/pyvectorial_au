#!/usr/bin/env python3

import numpy as np
import astropy.units as u
import scipy.interpolate

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib.colors import Normalize

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


def find_cdens_inflection_points(vmr: VectorialModelResult) -> np.ndarray:
    """
        Look for changes in sign of second derivative of the column density,
        given a VectorialModelResult and return a list of inflection points
    """

    xs = np.linspace(0, 5e8, num=100)
    concavity = vmr.column_density_interpolation.derivative(nu=2)
    ys = concavity(xs)

    # for pair in zip(xs, ys):
    #     print(f"R: {pair[0]:08.1e}\t\tConcavity: {pair[1]:8.8f}")

    # Array of 1s or 0s marking if the sign changed from one element to the next
    sign_changes = (np.diff(np.sign(ys)) != 0)*1

    # Manually remove the weirdness near the nucleus and rezize the array
    sign_changes[0] = 0
    sign_changes = np.resize(sign_changes, 100)

    inflection_points = xs*sign_changes
    # Only want non-zero elements
    inflection_points = inflection_points[inflection_points > 0]

    # Only want inflection points outside the collision sphere
    csphere_radius = vmr.collision_sphere_radius.to_value(u.m)
    inflection_points = inflection_points[inflection_points > csphere_radius]

    inflection_points = inflection_points * u.m
    return inflection_points


def mark_inflection_points(vmr: VectorialModelResult, ax, **kwargs) -> None:

    # Find possible inflection points
    for ipoint in find_cdens_inflection_points(vmr):
        ax.axvline(x=ipoint, **kwargs)


def mark_collision_sphere(vmr: VectorialModelResult, ax, **kwargs) -> None:

    # Mark the beginning of the collision sphere
    ax.axvline(x=vmr.collision_sphere_radius, **kwargs)


def volume_density_plot(vmr: VectorialModelResult, ax, r_units=u.m, vdens_units=1/u.m**3, **kwargs) -> None:

    xs = vmr.volume_density_grid.to(r_units)
    ys = vmr.volume_density.to(vdens_units)

    ax.scatter(xs, ys, **kwargs)


def volume_density_interpolation_plot(vmr: VectorialModelResult, ax, r_units=u.m, vdens_units=1/u.m**3, **kwargs) -> None:

    # model's interpolation function needs meters in, gives output in 1/m**3
    ys = vmr.volume_density_interpolation(vmr.volume_density_grid.to_value(u.m))/u.m**3
    ax.plot(vmr.volume_density_grid.to(r_units), ys.to(vdens_units), **kwargs)


def column_density_plot(vmr: VectorialModelResult, ax, r_units=u.m, cdens_units=1/u.m**2, **kwargs) -> None:

    xs = vmr.column_density_grid.to(r_units)
    ys = vmr.column_density.to(cdens_units)

    ax.scatter(xs, ys, **kwargs)


def column_density_interpolation_plot(vmr: VectorialModelResult, ax, r_units=u.m, cdens_units=1/u.m**2, **kwargs) -> None:

    # model's interpolation function needs meters in, gives output in 1/m**2
    ys = vmr.column_density_interpolation(vmr.column_density_grid.to_value(u.m))/u.m**2
    ax.plot(vmr.column_density_grid.to(r_units), ys.to(cdens_units), **kwargs)


def column_density_plot_3d(vmr: VectorialModelResult, ax, center=(0,0)*u.m, width=200000*u.km, height=200000*u.km, divisions=100, dist_units=u.m, cdens_units=1/u.m**2, **kwargs) -> None:

    xmin_m, ymin_m = np.subtract(center.to_value(u.m), (width.to_value(u.m)/2, height.to_value(u.m)/2))
    xmax_m, ymax_m = np.add((xmin_m, ymin_m), (width.to_value(u.m), height.to_value(u.m)))
    xs_m = np.linspace(xmin_m, xmax_m, num=divisions)
    ys_m = np.linspace(ymin_m, ymax_m, num=divisions)

    xmesh_m, ymesh_m = np.meshgrid(xs_m, ys_m)
    zmesh = vmr.column_density_interpolation(np.sqrt(xmesh_m**2 + ymesh_m**2))/u.m**2
    xmesh = xmesh_m * u.m
    ymesh = ymesh_m * u.m

    vmin = np.min(zmesh).to_value(cdens_units)

    ax.plot_surface(xmesh.to(dist_units), ymesh.to(dist_units), zmesh.to(cdens_units), vmin=vmin, **kwargs)
    ax.set_zlim(bottom=0)


def fragment_sputter_contour_plot(vmr, ax, dist_units=u.km, sputter_units=1/u.cm**3, within_r=1000*u.km, mirrored=False, show_outflow_axis=True, **kwargs) -> None:

    fsc = vmr.fragment_sputter

    if mirrored:
        fsc = mirror_sputter(fsc)

    if isinstance(fsc, FragmentSputterPolar):
        fsc = cartesian_sputter_from_polar(fsc)

    xs = fsc.xs.to(dist_units)
    ys = fsc.ys.to(dist_units)
    zs = fsc.fragment_density.to(sputter_units)

    within_limit = np.sqrt(xs**2 + ys**2) < within_r
    xs = xs[within_limit]
    ys = ys[within_limit]
    zs = zs[within_limit]

    if show_outflow_axis:
        # highlight the outflow axis, along positive y
        origin = [0, 0, 0] * dist_units
        outflow_max = [0, np.max(ys.to_value(dist_units)), 0] * dist_units
        ax.plot(origin, outflow_max, color=myblue, lw=2, label='outflow axis')

    x_mesh, y_mesh = np.meshgrid(np.unique(xs), np.unique(ys))
    fs_rbf = scipy.interpolate.Rbf(xs, ys, zs, function='cubic')
    frag_mesh = fs_rbf(x_mesh, y_mesh)

    ax.contourf(x_mesh, y_mesh, frag_mesh, levels=np.arange(np.min(frag_mesh), np.max(frag_mesh), 50), **kwargs)
    ax.contour(x_mesh, y_mesh, frag_mesh, levels=np.arange(np.min(frag_mesh), np.max(frag_mesh), 50), colors='black', linewidths=0.5)
    ax.set_aspect('equal')


def fragment_sputter_plot(vmr, ax, dist_units=u.m, sputter_units=1/u.m**3, within_r=1000*u.km, mirrored=False, show_outflow_axis=True, **kwargs) -> None:

    fsc = vmr.fragment_sputter

    if mirrored:
        fsc = mirror_sputter(fsc)

    if isinstance(fsc, FragmentSputterPolar):
        fsc = cartesian_sputter_from_polar(fsc)

    xs = fsc.xs.to(dist_units)
    ys = fsc.ys.to(dist_units)
    zs = fsc.fragment_density.to(sputter_units)

    within_limit = np.sqrt(xs**2 + ys**2) < within_r

    xs = xs[within_limit]
    ys = ys[within_limit]
    zs = zs[within_limit]

    colors_map = 'viridis'
    cm = plt.get_cmap(colors_map)
    cNorm = Normalize(vmin=np.min(zs.value), vmax=np.max(zs.value))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    # highlight the outflow axis, along positive y
    if show_outflow_axis:
        origin = [0, 0, 0] * dist_units
        outflow_max = [0, np.max(ys.to_value(dist_units)), 0] * dist_units
        ax.plot(origin, outflow_max, color=myblue, lw=2, label='outflow axis')

    ax.scatter(xs, ys, zs, c=kwargs.get('color', scalarMap.to_rgba(zs.value)))


# def old_column_density_plot_3d(vmc: VectorialModelConfig, vmr: VectorialModelResult,
#         x_min, x_max, y_min, y_max, grid_step_x, grid_step_y, r_units,
#         cd_units, view_angles=(90, 90), show_plots=True, out_file=None,
#         vmin=None, vmax=None):
#     """ Plot the column density in 2d around nucleus, with column density as z axis """
#
#     # TODO: this only supports python version
#
#     x = np.linspace(x_min.to(u.m).value, x_max.to(u.m).value, grid_step_x)
#     y = np.linspace(y_min.to(u.m).value, y_max.to(u.m).value, grid_step_y)
#     xv, yv = np.meshgrid(x, y)
#     z = vmr.column_density_interpolation(np.sqrt(xv**2 + yv**2))
#     # column_density_interpolation spits out m^-2
#     fz = (z/u.m**2).to(cd_units)
#
#     xu = np.linspace(x_min.to(r_units), x_max.to(r_units), grid_step_x)
#     yu = np.linspace(y_min.to(r_units), y_max.to(r_units), grid_step_y)
#     xvu, yvu = np.meshgrid(xu, yu)
#
#     plt.style.use('Solarize_Light2')
#     plt.style.use('dark_background')
#     plt.rcParams['grid.color'] = "black"
#
#     fig = plt.figure(figsize=(20, 20))
#     ax = plt.axes(projection='3d')
#     # ax.grid(False)
#     surf = ax.plot_surface(xvu, yvu, fz, cmap='inferno', vmin=vmin, vmax=vmax, edgecolor='none')
#
#     plt.gca().set_zlim(bottom=0)
#
#     ax.set_xlabel(f'Distance, ({r_units.to_string()})')
#     ax.set_ylabel(f'Distance, ({r_units.to_string()})')
#     ax.set_zlabel(f"Column density, {cd_units.unit.to_string()}")
#     plt.title(f"Calculated column density of {vmc.fragment.name}")
#
#     ax.w_xaxis.set_pane_color(solargreen)
#     ax.w_yaxis.set_pane_color(solarblue)
#     ax.w_zaxis.set_pane_color(solarblack)
#
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     ax.view_init(view_angles[0], view_angles[1])
#
#     if out_file is not None:
#         plt.savefig(out_file)
#     if show_plots:
#         plt.show()
#
#     return plt, fig, ax, surf

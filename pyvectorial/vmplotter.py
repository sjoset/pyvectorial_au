#!/usr/bin/env python3

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

from .vmconfig import VectorialModelConfig
from .vmresult import VectorialModelResult, FragmentSputterCartesian, FragmentSputterPolar, cartesian_sputter_from_polar, mirror_sputter

solarbluecol = np.array([38, 139, 220]) / 255.
solarblue = (solarbluecol[0], solarbluecol[1], solarbluecol[2], 1)
solargreencol = np.array([133, 153, 0]) / 255.
solargreen = (solargreencol[0], solargreencol[1], solargreencol[2], 1)
solarblackcol = np.array([0, 43, 54]) / 255.
solarblack = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)
solarwhitecol = np.array([238, 232, 213]) / 255.
solarwhite = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)


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

# TODO: make the plotters take an axis to draw on to separate styling the graphs
# and plotting the data
# use fig = plt.figure() and ax = fig.subplot, ax.plot()
# or TODO: make a stylesheet and load that in plt.style.use and do no styling in here at all


# def find_cdens_inflection_points(vmodel):
#     """
#         Look for changes in sign of second derivative of the column density,
#         given a vectorial model coma and return a list of inflection points
#     """
#
#     xs = np.linspace(0, 5e8, num=100)
#     concavity = vmodel['column_density_interpolation'].derivative(nu=2)
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
#     csphere_radius = vmodel['collision_sphere_radius'].to_value(u.m)
#     inflection_points = inflection_points[inflection_points > csphere_radius]
#
#     inflection_points = inflection_points * u.m
#     return inflection_points
#
#
# def radial_density_plots(vmodel, r_units, voldens_units, frag_name, show_plots=True, out_file=None):
#
#     interp_color = myblue
#     model_color = myred
#     linear_color = mygreen
#     csphere_color = mybblue
#     csphere_text_color = myblack
#     inflection_color = mybblack
#
#     # view of entire grid space around comet
#     x_min_logplot = 4
#     x_max_logplot = 11
#
#     # zoom-in near the nucleus
#     x_min_linear = (0 * u.km).to(u.m)
#     x_max_linear = (2000 * u.km).to(u.m)
#
#     lin_interp_x = np.linspace(x_min_linear.value, x_max_linear.value, num=200)
#     lin_interp_y = vmodel['r_dens_interpolation'](lin_interp_x)/(u.m**3)
#     lin_interp_x *= u.m
#
#     log_interp_x = np.logspace(x_min_logplot, x_max_logplot, num=200)
#     log_interp_y = vmodel['r_dens_interpolation'](log_interp_x)/(u.m**3)
#     log_interp_x *= u.m
#
#     plt.style.use('Solarize_Light2')
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#
#     ax1.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
#     ax1.set(ylabel=f"Fragment density, {voldens_units.unit.to_string()}")
#     ax2.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
#     ax2.set(ylabel=f"Fragment density, {voldens_units.unit.to_string()}")
#     fig.suptitle(f"Calculated radial volume density of {frag_name}")
#
#     ax1.set_xlim([x_min_linear.to(r_units), x_max_linear.to(r_units)])
#     ax1.plot(lin_interp_x.to(r_units), lin_interp_y.to(voldens_units), color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
#     ax1.plot(vmodel['radial_grid'].to(r_units), vmodel['radial_density'].to(voldens_units), 'o', color=model_color, label="model")
#     ax1.plot(vmodel['radial_grid'].to(r_units), vmodel['radial_density'].to(voldens_units), '--', color=linear_color,
#              linewidth=1.0, label="linear interpolation")
#
#     ax2.set_xscale('log')
#     ax2.set_yscale('log')
#     ax2.loglog(log_interp_x.to(r_units), log_interp_y.to(voldens_units), color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
#     ax2.loglog(vmodel['radial_grid'].to(r_units), vmodel['radial_density'].to(voldens_units), 'o',
#                color=model_color, label="model")
#     ax2.loglog(vmodel['radial_grid'].to(r_units), vmodel['radial_density'].to(voldens_units), '--',
#                color=linear_color, linewidth=1.0, label="linear interpolation")
#
#     ax1.set_ylim(bottom=0)
#
#     # Mark the beginning of the collision sphere
#     ax1.axvline(x=vmodel['collision_sphere_radius'], color=csphere_color)
#     ax2.axvline(x=vmodel['collision_sphere_radius'], color=csphere_color)
#
#     # Text for the collision sphere
#     plt.text((vmodel['collision_sphere_radius']*1.1).to(r_units), (lin_interp_y[0]/20).to(voldens_units), 'Collision Sphere Edge',
#              color=csphere_text_color)
#
#     plt.legend(loc='upper right', frameon=False)
#
#     # Find possible inflection points
#     for ipoint in find_cdens_inflection_points(vmodel):
#         ax1.axvline(x=ipoint, color=inflection_color)
#
#     if out_file is not None:
#         plt.savefig(out_file)
#     if show_plots:
#         plt.show()
#
#     return plt, fig, ax1, ax2
#
#
# def column_density_plots(vmodel, r_units, cd_units, frag_name, show_plots=True, out_file=None):
#
#     interp_color = myblue
#     model_color = myred
#     linear_color = mygreen
#     csphere_color = mybblue
#     csphere_text_color = myblack
#     inflection_color = mybblack
#
#     # in meters, typically the whole grid space covered here
#     x_min_logplot = 4
#     x_max_logplot = 11
#
#     # for a zoom-in near the nucleus
#     x_min_linear = (0 * u.km).to(u.m)
#     x_max_linear = (2000 * u.km).to(u.m)
#
#     # model works in meters
#     lin_interp_x = np.linspace(x_min_linear.value, x_max_linear.value, num=200)
#     lin_interp_y = vmodel['column_density_interpolation'](lin_interp_x)/(u.m**2)
#     lin_interp_x *= u.m
#
#     log_interp_x = np.logspace(x_min_logplot, x_max_logplot, num=200)
#     log_interp_y = vmodel['column_density_interpolation'](log_interp_x)/(u.m**2)
#     log_interp_x *= u.m
#
#     plt.style.use('Solarize_Light2')
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#
#     ax1.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
#     ax1.set(ylabel=f"Fragment column density, {cd_units.unit.to_string()}")
#     ax2.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
#     ax2.set(ylabel=f"Fragment column density, {cd_units.unit.to_string()}")
#     fig.suptitle(f"Calculated column density of {frag_name}")
#
#     ax1.set_xlim([x_min_linear.to(r_units), x_max_linear.to(r_units)])
#     ax1.plot(lin_interp_x.to(r_units), lin_interp_y.to(cd_units), color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
#     ax1.plot(vmodel['column_density_grid'].to(r_units), vmodel['column_densities'].to(cd_units), 'o', color=model_color, label="model")
#     ax1.plot(vmodel['column_density_grid'].to(r_units), vmodel['column_densities'].to(cd_units), '--', color=linear_color,
#              label="linear interpolation", linewidth=1.0)
#
#     ax2.set_xscale('log')
#     ax2.set_yscale('log')
#     ax2.loglog(log_interp_x.to(r_units), log_interp_y.to(cd_units), color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
#     ax2.loglog(vmodel['column_density_grid'].to(r_units), vmodel['column_densities'].to(cd_units), 'o', color=model_color, label="model")
#     ax2.loglog(vmodel['column_density_grid'].to(r_units), vmodel['column_densities'].to(cd_units), '--', color=linear_color,
#                label="linear interpolation", linewidth=1.0)
#
#     # limits for plot 1
#     ax1.set_ylim(bottom=0)
#
#     # Mark the beginning of the collision sphere
#     ax1.axvline(x=vmodel['collision_sphere_radius'], color=csphere_color)
#     ax2.axvline(x=vmodel['collision_sphere_radius'], color=csphere_color)
#
#     # Only plot as far as the maximum radius of our grid on log-log plot
#     ax2.axvline(x=vmodel['max_grid_radius'])
#
#     # Mark the collision sphere
#     plt.text((vmodel['collision_sphere_radius']*1.1).to(r_units), (lin_interp_y[0]/20).to(cd_units), 'Collision Sphere Edge',
#              color=csphere_text_color)
#
#     plt.legend(loc='upper right', frameon=False)
#
#     # Find possible inflection points
#     for ipoint in find_cdens_inflection_points(vmodel):
#         ax1.axvline(x=ipoint, color=inflection_color)
#         ax2.axvline(x=ipoint, color=inflection_color, linewidth=0.5)
#
#     if out_file is not None:
#         plt.savefig(out_file)
#     if show_plots:
#         plt.show()
#
#     return plt, fig, ax1, ax2
#
#
# def column_density_plot_3d(vmodel, x_min, x_max, y_min, y_max, grid_step_x, grid_step_y, r_units, cd_units,
#                            frag_name, view_angles=(90, 90), show_plots=True, out_file=None, vmin=None, vmax=None):
#
#     x = np.linspace(x_min.to(u.m).value, x_max.to(u.m).value, grid_step_x)
#     y = np.linspace(y_min.to(u.m).value, y_max.to(u.m).value, grid_step_y)
#     xv, yv = np.meshgrid(x, y)
#     z = vmodel['column_density_interpolation'](np.sqrt(xv**2 + yv**2))
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
#     plt.title(f"Calculated column density of {frag_name}")
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


# def build_sputter_fortran(sputter, within_r_km, mirrored=False):
#     """ Takes a sputter read from the fortran version's output and gives us easily plottable data """
#
#     # get values close to the nucleus - fortran distances are in km
#     sputter = np.array([x for x in sputter if x[0] < within_r_km])
#
#     rs = sputter[:, 0]
#     thetas = sputter[:, 1]
#     zs = sputter[:, 2]
#
#     xs = rs*np.sin(thetas)
#     ys = rs*np.cos(thetas)
#
#     if mirrored:
#         xs = np.append(xs, -1*xs)
#         ys = np.append(ys, ys)
#         zs = np.append(zs, zs)
#
#     return xs, ys, zs
#
#
# def plot_sputter_fortran(sputter, within_r_km, mirrored=False, trisurf=False, show_plots=True, out_file=None):
#     """ Do the actual plotting of the fragment sputter """
#
#     xs, ys, zs = build_sputter_fortran(sputter, within_r_km, mirrored=mirrored)
#
#     plt.style.use('Solarize_Light2')
#     fig = plt.figure(figsize=(20, 20))
#
#     colorsMap = 'jet'
#     cm = plt.get_cmap(colorsMap)
#     cNorm = Normalize(vmin=np.min(zs), vmax=np.max(zs))
#     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
#
#     ax = Axes3D(fig)
#     ax.set(xlabel="x, km")
#     ax.set(ylabel="y, km")
#     ax.set(zlabel="fragment volume density, 1/cm^2")
#     fig.suptitle("Fortran fragment sputter")
#
#     # highlight the outflow axis, along positive y
#     origin = [0, 0, 0]
#     outflow_max = [0, within_r_km, 0]
#     ax.plot(origin, outflow_max, color=myblue, lw=2, label='outflow axis')
#     # ax.quiver(0, within_r_km/2, 0, 0, within_r_km*0.4, 0, color=mybblue, lw=2)
#
#     if trisurf:
#         ax.plot_trisurf(xs, ys, zs, color='white', edgecolors='grey', alpha=0.5)
#     ax.scatter(xs, ys, zs, c=scalarMap.to_rgba(zs))
#     scalarMap.set_array(zs)
#     fig.colorbar(scalarMap)
#     plt.legend(loc='upper right', frameon=False)
#     if out_file is not None:
#         plt.savefig(out_file)
#     if show_plots:
#         plt.show()
#     plt.close(fig)
#
#
# def build_sputter_python(vmodel, within_r_km, mirrored=False):
#     """ Return plottable data from a finished python vectorial model run """
#
#     vm_rs = vmodel['fast_radial_grid']
#     vm_thetas = vmodel['angular_grid']
#
#     sputterlist = []
#     for (i, j), vdens in np.ndenumerate(vmodel['density_grid']):
#         sputterlist.append([vm_rs[i], vm_thetas[j], vdens])
#     sputter = np.array(sputterlist)
#
#     # get close to the nucleus, vectorial model uses meters internally
#     sputter = np.array([x for x in sputter if x[0] < within_r_km*1000])
#
#     rs = sputter[:, 0]
#     thetas = sputter[:, 1]
#     zs = sputter[:, 2]
#
#     xs = rs*np.sin(thetas)
#     ys = rs*np.cos(thetas)
#
#     if mirrored:
#         xs = np.append(xs, -1*xs)
#         ys = np.append(ys, ys)
#         zs = np.append(zs, zs)
#
#     return xs, ys, zs
#
#
# def plot_sputter_python(vmodel, within_r_km, mirrored=False, trisurf=False, show_plots=True, out_file=None):
#     """ Plot the sputter """
#
#     xs, ys, zs = build_sputter_python(vmodel, within_r_km, mirrored=mirrored)
#     # convert python distances to km from m
#     xs = xs/1000
#     ys = ys/1000
#     # convert python density to 1/cm**3 from 1/m**3
#     zs = zs/1e6
#
#     fig = plt.figure(figsize=(20, 20))
#
#     plt.style.use('Solarize_Light2')
#     colorsMap = 'viridis'
#     # colorsMap = 'jet'
#     cm = plt.get_cmap(colorsMap)
#
#     method = 'saturate'
#     if method == 'saturate':
#         cNorm = Normalize(vmin=np.min(zs), vmax=np.max(zs)/30)
#     # elif method == 'multiple_of_min':
#     #     # only color the points less than a threshold value to see the detail
#     #     # in the tinier sputter values
#     #     zmask = 100
#     #     zs = np.ma.masked_where(zs >= np.min(zs)*zmask, zs)
#     #     cNorm = Normalize(vmin=np.min(zs), vmax=zmask*np.min(zs))
#     # elif method == 'percent_of_max':
#     #     zs = np.ma.masked_where(zs <= np.max(zs)/10, zs)
#     #     cNorm = Normalize(vmin=np.min(zs), vmax=np.max(zs))
#
#     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
#
#     ax = Axes3D(fig)
#     ax.set(xlabel="x, km")
#     ax.set(ylabel="y, km")
#     ax.set(zlabel="fragment volume density, 1/cm^2")
#     fig.suptitle("Python fragment sputter")
#     ax.set_box_aspect((1, 1, 1))
#     # ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
#
#     # highlight the outflow axis, along positive y
#     origin = [0, 0, 0]
#     outflow_max = [0, within_r_km, 0]
#     ax.plot(origin, outflow_max, color=myblue, lw=2, label='outflow axis')
#     # ax.quiver(0, within_r_km/2, 0, 0, within_r_km*0.4, 0, color=mybblue, lw=2)
#
#     if trisurf:
#         ax.plot_trisurf(xs, ys, zs, color='white', edgecolors='grey', alpha=0.5)
#     ax.scatter(xs, ys, zs, c=scalarMap.to_rgba(zs))
#     scalarMap.set_array(zs)
#     fig.colorbar(scalarMap)
#     plt.legend(loc='upper right', frameon=False)
#     if out_file is not None:
#         plt.savefig(out_file)
#     if show_plots:
#         plt.show()
#     plt.close(fig)
#
#
# def plot_sputters(f_sputter, vmodel, within_r_km, mirrored=False, trisurf=False, show_plots=True, out_file=None):
#     """ Combined plotting of fortran and vectorial model results """
#
#     pxs, pys, pzs = build_sputter_python(vmodel, within_r_km, mirrored=mirrored)
#     # convert python distances to km from m
#     pxs = pxs/1000
#     pys = pys/1000
#     # convert python density to 1/cm**3 from 1/m**3
#     pzs = pzs/1e6
#     fxs, fys, fzs = build_sputter_fortran(f_sputter, within_r_km, mirrored=mirrored)
#
#     fig = plt.figure(figsize=(20, 20))
#     plt.style.use('Solarize_Light2')
#
#     colorsMap = 'viridis'
#     cm = plt.get_cmap(colorsMap)
#     cNorm = Normalize(vmin=np.min(pzs), vmax=np.max(pzs))
#     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
#
#     ax = Axes3D(fig)
#     ax.set(xlabel="x, km")
#     ax.set(ylabel="y, km")
#     ax.set(zlabel="fragment volume density, 1/cm^2")
#     fig.suptitle("Fragment sputter comparison")
#
#     # highlight the outflow axis, along positive y
#     outflow_max = 1000
#     origin = [0, 0, 0]
#     outflow_max = [0, within_r_km, 0]
#     ax.plot(origin, outflow_max, color=myblue, lw=2, label='outflow axis')
#     # ax.quiver(0, within_r_km/2, 0, 0, within_r_km*0.4, 0, color=mybblue, lw=2)
#
#     if trisurf:
#         # ax.plot_trisurf(fxs, fys, fzs, color='white', edgecolors='grey', alpha=0.5)
#         ax.plot_trisurf(pxs, pys, pzs, color='white', edgecolors='grey', alpha=0.5)
#     ax.scatter(pxs, pys, pzs, c=scalarMap.to_rgba(pzs), label='python')
#     # ax.scatter(fxs, fys, fzs, c=scalarMap.to_rgba(fzs))
#     ax.scatter(fxs, fys, fzs, color='red', label='fortran')
#     scalarMap.set_array(pzs)
#     plt.legend(loc='upper right', frameon=False)
#     fig.colorbar(scalarMap)
#     if out_file is not None:
#         plt.savefig(out_file)
#     if show_plots:
#         plt.show()
#     plt.close(fig)


def radial_density_plots_fortran(vmc: VectorialModelConfig, fvmr: VectorialModelResult, vmr: VectorialModelResult, show_plots=True, out_file=None):

    pymodel_color = myred
    linear_color = mygreen
    fmodel_color = mybblue

    x_min_linear = 0 * u.km
    x_max_linear = 10000 * u.km

    # lin_interp_x = vgrid
    lin_interp_x = fvmr.volume_density_grid
    # lin_interp_y = vdens
    lin_interp_y = fvmr.volume_density
    # lin_interp_x *= u.km

    plt.style.use('Solarize_Light2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set(xlabel="Distance from nucleus, km")
    ax1.set(ylabel="Fragment density, 1/cm^3")
    ax2.set(xlabel="Distance from nucleus, km")
    ax2.set(ylabel="Fragment density, 1/cm^3")
    fig.suptitle(f"Calculated radial volume density of {vmc.fragment.name}")

    ax1.set_xlim([x_min_linear, x_max_linear])
    ax1.plot(lin_interp_x, lin_interp_y, color=linear_color,  linewidth=2.0, linestyle="-", label="fortran linear interp")
    ax1.plot(lin_interp_x, lin_interp_y, 'o', color=fmodel_color, label="fortran model")
    ax1.plot(vmr.volume_density_grid.to(u.m), vmr.volume_density.to(1/u.cm**3), 'o', color=pymodel_color, label="python model")

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.loglog(lin_interp_x, lin_interp_y, color=linear_color,  linewidth=2.0, linestyle="-", label="fortran linear interp")
    ax2.loglog(lin_interp_x, lin_interp_y, 'o', label="fortran model", color=fmodel_color)
    ax2.loglog(vmr.volume_density_grid.to(u.m), vmr.volume_density.to(1/u.cm**3), 'o', color=pymodel_color, label="python model")

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0.1)

    plt.legend(loc='upper right', frameon=False)

    if out_file is not None:
        plt.savefig(out_file)
    if show_plots:
        plt.show()
    plt.close(fig)


def find_cdens_inflection_points(vmr: VectorialModelResult):
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


def radial_density_plots(vmc: VectorialModelConfig, vmr: VectorialModelResult, r_units, voldens_units, show_plots=True, out_file=None):

    interp_color = myblue
    model_color = myred
    linear_color = mygreen
    csphere_color = mybblue
    csphere_text_color = myblack
    inflection_color = mybblack

    # view of entire grid space around comet
    x_min_logplot = 4
    x_max_logplot = 11

    # zoom-in near the nucleus
    x_min_linear = (0 * u.km).to(u.m)
    x_max_linear = (2000 * u.km).to(u.m)

    if vmr.volume_density_interpolation:
        lin_interp_x = np.linspace(x_min_linear.value, x_max_linear.value, num=200)
        lin_interp_y = vmr.volume_density_interpolation(lin_interp_x)/(u.m**3)
        lin_interp_x *= u.m

        log_interp_x = np.logspace(x_min_logplot, x_max_logplot, num=200)
        log_interp_y = vmr.volume_density_interpolation(log_interp_x)/(u.m**3)
        log_interp_x *= u.m

    plt.style.use('Solarize_Light2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax1.set(ylabel=f"Fragment density, {voldens_units.unit.to_string()}")
    ax2.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax2.set(ylabel=f"Fragment density, {voldens_units.unit.to_string()}")
    fig.suptitle(f"Calculated radial volume density of {vmc.fragment.name}")

    ax1.set_xlim([x_min_linear.to(r_units), x_max_linear.to(r_units)])
    if vmr.volume_density_interpolation:
        ax1.plot(lin_interp_x.to(r_units), lin_interp_y.to(voldens_units), color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
    ax1.plot(vmr.volume_density_grid.to(r_units), vmr.volume_density.to(voldens_units), 'o', color=model_color, label="model")
    ax1.plot(vmr.volume_density_grid.to(r_units), vmr.volume_density.to(voldens_units), '--', color=linear_color,
             linewidth=1.0, label="linear interpolation")

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    if vmr.volume_density_interpolation:
        ax2.loglog(log_interp_x.to(r_units), log_interp_y.to(voldens_units), color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
    ax2.loglog(vmr.volume_density_grid.to(r_units), vmr.volume_density.to(voldens_units), 'o',
               color=model_color, label="model")
    ax2.loglog(vmr.volume_density_grid.to(r_units), vmr.volume_density.to(voldens_units), '--',
               color=linear_color, linewidth=1.0, label="linear interpolation")

    ax1.set_ylim(bottom=0)

    # Mark the beginning of the collision sphere
    ax1.axvline(x=vmr.collision_sphere_radius, color=csphere_color)
    ax2.axvline(x=vmr.collision_sphere_radius, color=csphere_color)

    # Text for the collision sphere
    plt.text((vmr.collision_sphere_radius*1.1).to(r_units), (vmr.volume_density[0]/20).to(voldens_units), 'Collision Sphere Edge',
             color=csphere_text_color)

    plt.legend(loc='upper right', frameon=False)

    # Find possible inflection points
    if vmr.volume_density_interpolation:
        for ipoint in find_cdens_inflection_points(vmr):
            ax1.axvline(x=ipoint, color=inflection_color)

    if out_file is not None:
        plt.savefig(out_file)
    if show_plots:
        plt.show()

    return plt, fig, ax1, ax2


def column_density_plots(vmc: VectorialModelConfig, vmr: VectorialModelResult, r_units, cd_units, show_plots=True, out_file=None):

    interp_color = myblue
    model_color = myred
    linear_color = mygreen
    csphere_color = mybblue
    csphere_text_color = myblack
    inflection_color = mybblack

    # in meters, typically the whole grid space covered here
    x_min_logplot = 4
    x_max_logplot = 11

    # for a zoom-in near the nucleus
    x_min_linear = (0 * u.km).to(u.m)
    x_max_linear = (2000 * u.km).to(u.m)

    if vmr.column_density_interpolation:
        # model works in meters
        lin_interp_x = np.linspace(x_min_linear.value, x_max_linear.value, num=200)
        lin_interp_y = vmr.column_density_interpolation(lin_interp_x)/(u.m**2)
        lin_interp_x *= u.m

        log_interp_x = np.logspace(x_min_logplot, x_max_logplot, num=200)
        log_interp_y = vmr.column_density_interpolation(log_interp_x)/(u.m**2)
        log_interp_x *= u.m

    plt.style.use('Solarize_Light2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax1.set(ylabel=f"Fragment column density, {cd_units.unit.to_string()}")
    ax2.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax2.set(ylabel=f"Fragment column density, {cd_units.unit.to_string()}")
    fig.suptitle(f"Calculated column density of {vmc.fragment.name}")

    ax1.set_xlim([x_min_linear.to(r_units), x_max_linear.to(r_units)])
    if vmr.column_density_interpolation:
        ax1.plot(lin_interp_x.to(r_units), lin_interp_y.to(cd_units), color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
    ax1.plot(vmr.column_density_grid.to(r_units), vmr.column_density.to(cd_units), 'o', color=model_color, label="model")
    ax1.plot(vmr.column_density_grid.to(r_units), vmr.column_density.to(cd_units), '--', color=linear_color,
             label="linear interpolation", linewidth=1.0)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    if vmr.column_density_interpolation:
        ax2.loglog(log_interp_x.to(r_units), log_interp_y.to(cd_units), color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
    ax2.loglog(vmr.column_density_grid.to(r_units), vmr.column_density.to(cd_units), 'o', color=model_color, label="model")
    ax2.loglog(vmr.column_density_grid.to(r_units), vmr.column_density.to(cd_units), '--', color=linear_color,
               label="linear interpolation", linewidth=1.0)

    # limits for plot 1
    ax1.set_ylim(bottom=0)

    # Mark the beginning of the collision sphere
    ax1.axvline(x=vmr.collision_sphere_radius, color=csphere_color)
    ax2.axvline(x=vmr.collision_sphere_radius, color=csphere_color)

    # Only plot as far as the maximum radius of our grid on log-log plot
    ax2.axvline(x=vmr.max_grid_radius)

    # Mark the collision sphere
    plt.text((vmr.collision_sphere_radius*1.1).to(r_units), (vmr.column_density[0]/20).to(cd_units), 'Collision Sphere Edge',
             color=csphere_text_color)

    plt.legend(loc='upper right', frameon=False)

    if vmr.column_density_interpolation:
        # Find possible inflection points
        for ipoint in find_cdens_inflection_points(vmr):
            ax1.axvline(x=ipoint, color=inflection_color)
            ax2.axvline(x=ipoint, color=inflection_color, linewidth=0.5)

    if out_file is not None:
        plt.savefig(out_file)
    if show_plots:
        plt.show()

    return plt, fig, ax1, ax2


def column_density_plot_3d(vmc: VectorialModelConfig, vmr: VectorialModelResult,
        x_min, x_max, y_min, y_max, grid_step_x, grid_step_y, r_units,
        cd_units, view_angles=(90, 90), show_plots=True, out_file=None,
        vmin=None, vmax=None):

    # TODO: this only support python version

    x = np.linspace(x_min.to(u.m).value, x_max.to(u.m).value, grid_step_x)
    y = np.linspace(y_min.to(u.m).value, y_max.to(u.m).value, grid_step_y)
    xv, yv = np.meshgrid(x, y)
    z = vmr.column_density_interpolation(np.sqrt(xv**2 + yv**2))
    # column_density_interpolation spits out m^-2
    fz = (z/u.m**2).to(cd_units)

    xu = np.linspace(x_min.to(r_units), x_max.to(r_units), grid_step_x)
    yu = np.linspace(y_min.to(r_units), y_max.to(r_units), grid_step_y)
    xvu, yvu = np.meshgrid(xu, yu)

    plt.style.use('Solarize_Light2')
    plt.style.use('dark_background')
    plt.rcParams['grid.color'] = "black"

    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')
    # ax.grid(False)
    surf = ax.plot_surface(xvu, yvu, fz, cmap='inferno', vmin=vmin, vmax=vmax, edgecolor='none')

    plt.gca().set_zlim(bottom=0)

    ax.set_xlabel(f'Distance, ({r_units.to_string()})')
    ax.set_ylabel(f'Distance, ({r_units.to_string()})')
    ax.set_zlabel(f"Column density, {cd_units.unit.to_string()}")
    plt.title(f"Calculated column density of {vmc.fragment.name}")

    ax.w_xaxis.set_pane_color(solargreen)
    ax.w_yaxis.set_pane_color(solarblue)
    ax.w_zaxis.set_pane_color(solarblack)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(view_angles[0], view_angles[1])

    if out_file is not None:
        plt.savefig(out_file)
    if show_plots:
        plt.show()

    return plt, fig, ax, surf


def plot_fragment_sputter(fsc, dist_units, sputter_units, within_r=None, trisurf=False, mirrored=False, show_plots=True, out_file=None):

    if mirrored:
        fsc = mirror_sputter(fsc)

    if isinstance(fsc, FragmentSputterPolar):
        fsc = cartesian_sputter_from_polar(fsc)

    xs = fsc.xs.to(dist_units)
    ys = fsc.ys.to(dist_units)
    zs = fsc.fragment_density.to(sputter_units)

    within_limit = np.sqrt(xs**2 + ys**2) < (within_r.to(dist_units))

    xs = xs[within_limit]
    ys = ys[within_limit]
    zs = zs[within_limit]

    fig = plt.figure(figsize=(20, 20))

    plt.style.use('Solarize_Light2')
    colors_map = 'viridis'
    # colorsMap = 'jet'
    cm = plt.get_cmap(colors_map)

    method = 'saturate'
    if method == 'saturate':
        cNorm = Normalize(vmin=np.min(zs.value), vmax=np.max(zs.value))
        # cNorm = Normalize(vmin=np.min(zs.value), vmax=np.max(zs.value)/30)
    # elif method == 'multiple_of_min':
    #     # only color the points less than a threshold value to see the detail
    #     # in the tinier sputter values
    #     zmask = 100
    #     zs = np.ma.masked_where(zs >= np.min(zs)*zmask, zs)
    #     cNorm = Normalize(vmin=np.min(zs), vmax=zmask*np.min(zs))
    # elif method == 'percent_of_max':
    #     zs = np.ma.masked_where(zs <= np.max(zs)/10, zs)
    #     cNorm = Normalize(vmin=np.min(zs), vmax=np.max(zs))

    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    ax = Axes3D(fig)
    ax.set(xlabel=f"x, {dist_units.to_string()}")
    ax.set(ylabel=f"y, {dist_units.to_string()}")
    ax.set(zlabel=f"fragment volume density, {sputter_units.unit.to_string()}")
    fig.suptitle("Fragment sputter")
    ax.set_box_aspect((1, 1, 1))
    # ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))

    # highlight the outflow axis, along positive y
    origin = [0, 0, 0]
    outflow_max = [0, within_r.to(dist_units).value, 0]
    ax.plot(origin, outflow_max, color=myblue, lw=2, label='outflow axis')
    # ax.quiver(0, within_r_km/2, 0, 0, within_r_km*0.4, 0, color=mybblue, lw=2)

    if trisurf:
        ax.plot_trisurf(xs, ys, zs, color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(xs, ys, zs, c=scalarMap.to_rgba(zs.value))

    scalarMap.set_array(fsc.fragment_density.to(sputter_units).value)
    # fig.colorbar(scalarMap)
    plt.legend(loc='upper right', frameon=False)
    if out_file is not None:
        plt.savefig(out_file)
    if show_plots:
        plt.show()

    return plt, fig, ax


def plot_sputters(fortran_sputter: FragmentSputterCartesian, sbpy_sputter: FragmentSputterCartesian, dist_units, sputter_units, within_r=None, trisurf=False, mirrored=False, show_plots=True, out_file=None):

    if mirrored:
        fortran_sputter = mirror_sputter(fortran_sputter)
        sbpy_sputter = mirror_sputter(sbpy_sputter)

    if isinstance(fortran_sputter, FragmentSputterPolar):
        fortran_sputter = cartesian_sputter_from_polar(fortran_sputter)
    if isinstance(sbpy_sputter, FragmentSputterPolar):
        sbpy_sputter = cartesian_sputter_from_polar(sbpy_sputter)

    fxs = fortran_sputter.xs.to(dist_units)
    fys = fortran_sputter.ys.to(dist_units)
    fzs = fortran_sputter.fragment_density.to(sputter_units)

    fwithin_limit = np.sqrt(fxs**2 + fys**2) < (within_r.to(dist_units))

    fxs = fxs[fwithin_limit]
    fys = fys[fwithin_limit]
    fzs = fzs[fwithin_limit]

    xs = sbpy_sputter.xs.to(dist_units)
    ys = sbpy_sputter.ys.to(dist_units)
    zs = sbpy_sputter.fragment_density.to(sputter_units)

    within_limit = np.sqrt(xs**2 + ys**2) < (within_r.to(dist_units))

    xs = xs[within_limit]
    ys = ys[within_limit]
    zs = zs[within_limit]

    fig = plt.figure(figsize=(20, 20))

    plt.style.use('Solarize_Light2')
    colorsMap = 'viridis'
    # colorsMap = 'jet'
    cm = plt.get_cmap(colorsMap)

    method = 'saturate'
    if method == 'saturate':
        cNorm = Normalize(vmin=np.min(sbpy_sputter.fragment_density.to(sputter_units).value), vmax=np.max(sbpy_sputter.fragment_density.to(sputter_units).value)/30)
    # elif method == 'multiple_of_min':
    #     # only color the points less than a threshold value to see the detail
    #     # in the tinier sputter values
    #     zmask = 100
    #     zs = np.ma.masked_where(zs >= np.min(zs)*zmask, zs)
    #     cNorm = Normalize(vmin=np.min(zs), vmax=zmask*np.min(zs))
    # elif method == 'percent_of_max':
    #     zs = np.ma.masked_where(zs <= np.max(zs)/10, zs)
    #     cNorm = Normalize(vmin=np.min(zs), vmax=np.max(zs))

    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    ax = Axes3D(fig)
    ax.set(xlabel=f"x, {dist_units.to_string()}")
    ax.set(ylabel=f"y, {dist_units.to_string()}")
    ax.set(zlabel=f"fragment volume density, {sputter_units.unit.to_string()}")
    fig.suptitle("Fragment sputter comparison")
    ax.set_box_aspect((1, 1, 1))
    # ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))

    # highlight the outflow axis, along positive y
    origin = [0, 0, 0]
    outflow_max = [0, within_r.to(dist_units).value, 0]
    ax.plot(origin, outflow_max, color=myblue, lw=2, label='outflow axis')
    # ax.quiver(0, within_r_km/2, 0, 0, within_r_km*0.4, 0, color=mybblue, lw=2)

    if trisurf:
        ax.plot_trisurf(sbpy_sputter.xs.to(dist_units), sbpy_sputter.ys.to(dist_units), sbpy_sputter.fragment_density.to(sputter_units), color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(xs, ys, zs, c=scalarMap.to_rgba(zs.value))
    ax.scatter(fxs, fys, fzs, color='red', label='fortran')
    scalarMap.set_array(sbpy_sputter.fragment_density.to(sputter_units).value)
    # fig.colorbar(scalarMap)
    plt.legend(loc='upper right', frameon=False)
    if out_file is not None:
        plt.savefig(out_file)
    if show_plots:
        plt.show()
    plt.close(fig)


# def plot_sputter_python_interpolated(vmodel, within_r_km):
#
#     # TODO: figure out how to make this less ugly and actually work
#     xs, ys, zs = pyv.build_sputter_python(vmodel, within_r_km, mirrored=False)
#     xs = xs/1e3
#     ys = ys/1e3
#     zs /= 1e9
#     # xs, ys, zs = build_sputter_fortran(f_sputter)
#
#     xi = np.logspace(-1, 2, 700)
#     yi = np.logspace(-1, 2, 700)
#     # xi = np.linspace(0, 100, 500)
#     # yi = np.linspace(-100, 400, 1000)
#     xi, yi = np.meshgrid(xi, yi)
#
#     zi = griddata((xs, ys), zs, (xi, yi), method='cubic')
#     plt.figure()
#     plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
#     plt.contourf(xi, yi, zi, 500, cmap='magma')
#     plt.colorbar()
#     plt.show()

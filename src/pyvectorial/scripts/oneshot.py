#!/usr/bin/env python3

import os
import sys
import pathlib
import logging as log
from argparse import ArgumentParser
from typing import Union

import numpy as np
import pandas as pd
import astropy.units as u
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import pyvectorial as pyv
from pyvectorial.backends.python_version import PythonModelExtraConfig
from pyvectorial.backends.rust_version import RustModelExtraConfig
from pyvectorial.backends.fortran_version import FortranModelExtraConfig
from pyvectorial.encoding_and_hashing import vmc_to_sha256_digest
from pyvectorial.graphing.vm_matplotlib import (
    mpl_column_density_interpolation_plot,
    mpl_column_density_plot,
    mpl_column_density_plot_3d,
    mpl_fragment_sputter_contour_plot,
    mpl_fragment_sputter_plot,
    mpl_volume_density_interpolation_plot,
    mpl_volume_density_plot,
)
from pyvectorial.graphing.vm_plotly import (
    plotly_column_density_interpolation_plot,
    plotly_column_density_plot,
    plotly_fragment_sputter_contour_plot,
    plotly_fragment_sputter_plot,
    plotly_volume_density_interpolation_plot,
    plotly_volume_density_plot,
)
from pyvectorial.input_transforms import VmcTransform, apply_input_transform
from pyvectorial.vectorial_model_calculation import (
    dataframe_to_vmcalc_list,
    load_vmcalculation_list,
    store_vmcalculation_list,
    vmcalc_list_to_dataframe,
)


model_backend_configs = {
    "sbpy (python)": PythonModelExtraConfig(),
    "rustvec (rust)": RustModelExtraConfig(),
    "vm (fortran)": FortranModelExtraConfig(
        fortran_input_filename=pathlib.Path("fparam.dat"),
        fortran_output_filename=pathlib.Path("fort.16"),
        r_h=1.0 * u.AU,  # type: ignore
    ),
}


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile] [outputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    # parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "parameterfile", nargs=1, help="YAML file with production and molecule data"
    )  # the nargs=? specifies 0 or 1 arguments: it is optional
    # parser.add_argument("output_fits", nargs=1, help="Filename of FITS output")

    args = parser.parse_args()

    # handle verbosity
    if args.verbose >= 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif args.verbose == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return args


def get_backend_model_selection() -> (
    Union[
        PythonModelExtraConfig,
        FortranModelExtraConfig,
        RustModelExtraConfig,
    ]
):
    backend_selection = None
    model_backend_names = list(model_backend_configs.keys())

    while backend_selection is None:
        print("Select a model backend:")
        for i, dataset_name in enumerate(model_backend_names):
            print(f"{i}:\t{dataset_name}")

        raw_selection = input()
        try:
            selection = int(raw_selection)
        except ValueError:
            print("Numbers only, please")
            selection = -1

        if selection in range(len(model_backend_names)):
            backend_selection = model_backend_names[selection]

    return model_backend_configs[backend_selection]


def fragment_sputter_contour_plot_plotly(vmr: pyv.VectorialModelResult):
    myblue = "#688894"

    fig = go.Figure()

    sputter, outflow, max_coord = plotly_fragment_sputter_contour_plot(
        vmr,
        dist_units=u.km,
        sputter_units=1 / u.cm**3,
        within_r=10000 * u.km,  # type: ignore
        min_r=1000 * u.km,  # type: ignore
        max_angle=np.pi / 16,
        mirrored=True,
    )
    fig.add_trace(sputter)
    if outflow:
        fig.add_trace(outflow)

    fig.data[1].line.color = myblue  # type: ignore

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=[-max_coord, max_coord],
            ),
            yaxis=dict(
                nticks=4,
                range=[-max_coord, max_coord],
            ),
            # zaxis = dict(nticks=4, range=[-100,100],),
            xaxis_title="aoeu",
        ),
        # width=1800,
        margin=dict(r=20, l=10, b=10, t=10),
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.show()


def fragment_sputter_plot_plotly(vmr: pyv.VectorialModelResult):
    myblue = "#688894"
    fig = go.Figure()

    sputter, outflow, max_coord = plotly_fragment_sputter_plot(
        vmr,
        dist_units=u.km,
        sputter_units=1 / u.cm**3,
        within_r=15000 * u.km,  # type: ignore
        mirrored=True,
        marker_colorscale="Viridis",
    )
    fig.add_trace(sputter)
    if outflow:
        fig.add_trace(outflow)

    fig.data[1].line.color = myblue  # type: ignore
    max_coord *= 1.1  # type: ignore

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=[-max_coord, max_coord],
            ),
            yaxis=dict(
                nticks=4,
                range=[-max_coord, max_coord],
            ),
            # zaxis = dict(nticks=4, range=[-100,100],),
            xaxis_title="aoeu",
        ),
        # width=1800,
        xaxis_range=[-max_coord, max_coord],
        yaxis_range=[-500, 3000],
        margin=dict(r=20, l=10, b=10, t=10),
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.show()


def volume_and_column_density_plots_plotly(vmr: pyv.VectorialModelResult):
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        plotly_volume_density_plot(
            vmr, dist_units=u.km, vdens_units=1 / u.cm**3, opacity=0.5, mode="markers"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        plotly_volume_density_interpolation_plot(
            vmr, dist_units=u.km, vdens_units=1 / u.cm**3, mode="lines"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        plotly_column_density_plot(
            vmr, dist_units=u.km, cdens_units=1 / u.cm**2, opacity=0.5, mode="markers"
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        plotly_column_density_interpolation_plot(
            vmr, dist_units=u.km, cdens_units=1 / u.cm**2, mode="lines"
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=[-100, 100],
            ),
            yaxis=dict(
                nticks=4,
                range=[-50, 100],
            ),
            zaxis=dict(
                nticks=4,
                range=[-100, 100],
            ),
            xaxis_title="radius",
        ),
        width=1800,
        margin=dict(r=20, l=10, b=10, t=10),
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    fig.show()


def volume_and_column_density_plots_mpl(vmr: pyv.VectorialModelResult):
    _, axs = plt.subplots(1, 2, sharex=True, layout="constrained")
    mpl_volume_density_plot(vmr=vmr, ax=axs[0], dist_units=u.km, alpha=0.5)
    mpl_volume_density_interpolation_plot(vmr=vmr, ax=axs[0], dist_units=u.km)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    mpl_column_density_plot(vmr=vmr, ax=axs[1], dist_units=u.km, alpha=0.5)
    mpl_column_density_interpolation_plot(vmr=vmr, dist_units=u.km, ax=axs[1])
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    plt.show()


def fragment_sputter_plot_mpl(vmr: pyv.VectorialModelResult):
    _, axs = plt.subplots(1, 2, layout="constrained", subplot_kw={"projection": "3d"})
    mpl_fragment_sputter_plot(vmr=vmr, ax=axs[0], within_r=2000 * u.km)  # type: ignore
    mpl_column_density_plot_3d(vmr=vmr, ax=axs[1])
    plt.show()


def fragment_sputter_contour_plot_mpl(vmr: pyv.VectorialModelResult):
    _, axs = plt.subplots(1, 1, layout="constrained")
    mpl_fragment_sputter_contour_plot(
        vmr=vmr, ax=axs, within_r=400 * u.km, max_angle=np.pi / 8, mirrored=True  # type: ignore
    )
    plt.show()


def main():
    args = process_args()
    log.debug("Loading input from %s ....", args.parameterfile[0])

    vmc_unxfrmed = pyv.vectorial_model_config_from_yaml(
        pathlib.Path(args.parameterfile[0])
    )
    if vmc_unxfrmed is None:
        print(f"Failed to read {args.parameterfile}!")
        return 1

    # r_h = 5.92 * u.AU  # type: ignore
    r_h = 2.0 * u.AU  # type: ignore
    vmc = apply_input_transform(
        vmc=vmc_unxfrmed, r_h=r_h, xfrm=VmcTransform.cochran_schleicher_93
    )

    vmc_set = [vmc, vmc_unxfrmed, vmc, vmc_unxfrmed, vmc, vmc_unxfrmed]

    ec = get_backend_model_selection()

    vmcalc_list = pyv.run_vectorial_models_pooled(
        vmc_set=vmc_set, extra_config=ec, parallelism=2
    )

    result_storage_path_stem = pathlib.Path(args.parameterfile[0]).stem
    result_storage_path = pathlib.Path(result_storage_path_stem).with_suffix(".vmcl")

    store_vmcalculation_list(vmcalc_list, result_storage_path)
    vmcp_new = load_vmcalculation_list(result_storage_path)

    for vmcalc in vmcp_new:
        print(vmc_to_sha256_digest(vmcalc.vmc))

    df = vmcalc_list_to_dataframe(vmcp_new)
    print(df)
    new_vmcalc_list = dataframe_to_vmcalc_list(df)
    print(new_vmcalc_list[0].vmr.coma_radius)
    for v, d in zip(vmcalc_list, df["vmc_sha256_digest"]):
        print(f"{v.execution_time_s}\tsha256: {d}")

    df.to_csv("dfout.csv", index=False)

    new_df = pd.read_csv("dfout.csv")
    print(new_df["vmc_sha256_digest"])

    nvmcl = dataframe_to_vmcalc_list(new_df)
    for x in nvmcl:
        print(
            f" {x.vmr.coma_radius=}, {x.vmr.max_grid_radius=}, {x.vmr.max_grid_radius / x.vmr.coma_radius}"
        )

    # volume_and_column_density_plots_plotly(vmr=nvmcl[0].vmr)
    # volume_and_column_density_plots_plotly(vmr=vmcp_new[0].vmr)
    # fragment_sputter_plot_plotly(vmr=nvmcl[0].vmr)
    # fragment_sputter_contour_plot_plotly(vmr=vmr)

    # volume_and_column_density_plots_mpl(vmr=vmr)
    # fragment_sputter_plot_mpl(vmr=vmr)
    # fragment_sputter_contour_plot_mpl(vmr=vmr)


if __name__ == "__main__":
    sys.exit(main())

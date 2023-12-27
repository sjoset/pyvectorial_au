#!/usr/bin/env python3

import os
import sys
import pathlib
import contextlib
import hashlib
import logging as log
from argparse import ArgumentParser
from typing import Union

import numpy as np
import astropy.units as u
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import pyvectorial as pyv
from pyvectorial.backends.python_version import PythonModelExtraConfig
from pyvectorial.backends.rust_version import RustModelExtraConfig
from pyvectorial.backends.fortran_version import FortranModelExtraConfig
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
from pyvectorial.vectorial_model_config import hash_vmc


rustbin = pathlib.Path(
    pathlib.Path.home(),
    pathlib.Path(
        "repos/aucomet/vectorial_model/src/model_language_comparison/bin/rust_vect"
    ),
)
fortranbin = pathlib.Path(
    pathlib.Path.home(),
    pathlib.Path("repos/aucomet/vectorial_model/src/model_language_comparison/bin/fvm"),
)

model_backend_configs = {
    "sbpy (python)": PythonModelExtraConfig(print_progress=True),
    "rustvec (rust)": RustModelExtraConfig(
        rust_input_filename=pathlib.Path("rust_in.yaml"),
        rust_output_filename=pathlib.Path("rust_out.txt"),
    ),
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


def remove_file_silent_fail(f: pathlib.Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(f)


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
        within_r=10000 * u.km,
        min_r=1000 * u.km,
        max_angle=np.pi / 16,
        mirrored=True,
    )
    fig.add_trace(sputter)
    if outflow:
        fig.add_trace(outflow)

    fig.data[1].line.color = myblue

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
        within_r=15 * u.km,
        mirrored=True,
        marker_colorscale="Viridis",
    )
    fig.add_trace(sputter)
    if outflow:
        fig.add_trace(outflow)

    fig.data[1].line.color = myblue
    max_coord *= 1.1

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
    fig, axs = plt.subplots(1, 2, sharex=True, layout="constrained")
    # mpl_fragment_sputter_plot(vmr, axs[0])
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
    fig, axs = plt.subplots(1, 2, layout="constrained", subplot_kw={"projection": "3d"})
    mpl_fragment_sputter_plot(vmr=vmr, ax=axs[0], within_r=2000 * u.km)
    mpl_column_density_plot_3d(vmr=vmr, ax=axs[1])
    plt.show()


def fragment_sputter_contour_plot_mpl(vmr: pyv.VectorialModelResult):
    fig, axs = plt.subplots(1, 1, layout="constrained")
    mpl_fragment_sputter_contour_plot(
        vmr=vmr, ax=axs, within_r=400 * u.km, max_angle=np.pi / 8, mirrored=True
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
    r_h = 1.0 * u.AU  # type: ignore
    vmc = apply_input_transform(
        vmc=vmc_unxfrmed, r_h=r_h, xfrm=VmcTransform.cochran_schleicher_93
    )
    print(hash_vmc(vmc))
    print(hash(vmc))
    m = hashlib.sha256()
    m.update(vmc.model_dump_json().encode())
    print(m.hexdigest())

    vmc_set = [vmc]

    ec = get_backend_model_selection()

    out_table = pyv.build_calculation_table(vmc_set, extra_config=ec)  # type: ignore
    vmr = pyv.unpickle_from_base64(out_table["b64_encoded_vmr"][0])  # type: ignore

    print(hash_vmc(vmc))
    print(hash(vmc))
    n = hashlib.sha256()
    n.update(vmc.model_dump_json().encode())
    print(n.hexdigest())

    vmcalc_list = pyv.run_vectorial_models_pooled(
        vmc_set=vmc_set, extra_config=ec, parallelism=2
    )
    print(vmcalc_list)

    # volume_and_column_density_plots_plotly(vmr=vmr)
    # fragment_sputter_plot_plotly(vmr=vmr)
    # fragment_sputter_contour_plot_plotly(vmr=vmr)

    # volume_and_column_density_plots_mpl(vmr=vmr)
    # fragment_sputter_plot_mpl(vmr=vmr)
    # fragment_sputter_contour_plot_mpl(vmr=vmr)

    # print(f"vmc dict: {vmc.dict()}")
    # vmc_json = vmc.model_dump_json()
    # print(f"vmc json: {vmc_json}")
    #
    # with open("vmc.json", "w") as f:
    #     json.dump(vmc_json, f)
    #
    # with open("vmc.json", "r") as f:
    #     vmc_read_json = json.load(f)
    #
    # print(vmc_read_json)
    # vmc_read = VectorialModelConfig.model_validate_json(vmc_read_json)
    # print(vmc_read)
    #
    # new_out_table = pyv.build_calculation_table([vmc_read], extra_config=ec)
    # new_vmr = pyv.unpickle_from_base64(out_table["b64_encoded_vmr"][0])
    # print(new_vmr)

    # r_kms = vmr.column_density_grid.to_value(u.km)
    # cds = vmr.column_density.to_value(1 / u.cm**2)
    # df = pd.DataFrame({"r_kms": r_kms, "column_density_per_cm2": cds})
    #
    # output_dir = pathlib.Path("output")
    # output_file_stem = pathlib.Path(args.parameterfile[0]).stem
    # output_file_name = pathlib.Path(output_file_stem + ".fits")
    # output_file_path = output_dir / output_file_name
    # # output_fits_file = pathlib.Path(args.output_fits[0])
    # remove_file_silent_fail(output_file_path)
    # log.info("Table building complete, writing results to %s ...", output_file_path)
    # out_table.write(output_file_path, format="fits")
    #
    # df.to_csv(output_file_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())

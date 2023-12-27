#!/usr/bin/env python3

import os
import sys
import pathlib
import contextlib
import logging as log
from argparse import ArgumentParser
from typing import Union

import numpy as np
import astropy.units as u
import plotly.graph_objects as go
from rich import print as rprint

import pyvectorial as pyv
from pyvectorial.backends.python_version import PythonModelExtraConfig
from pyvectorial.backends.rust_version import RustModelExtraConfig
from pyvectorial.backends.fortran_version import FortranModelExtraConfig
from pyvectorial.graphing.vm_plotly import (
    plotly_fragment_sputter_plot,
)
from pyvectorial.input_transforms import VmcTransform, apply_input_transform
from pyvectorial.vectorial_model_result import VectorialModelResult


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


def fragment_sputter_plot(vmr: pyv.VectorialModelResult):
    myblue = "#688894"
    fig = go.Figure()

    sputter, outflow, max_coord = plotly_fragment_sputter_plot(
        vmr,
        dist_units=u.km,
        sputter_units=1 / u.cm**3,
        within_r=3000 * u.km,
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


def model_compare(vmr_one: VectorialModelResult, vmr_two: VectorialModelResult) -> None:
    if (
        vmr_one.volume_density_interpolation is None
        or vmr_two.volume_density_interpolation is None
    ):
        return

    vmr1_min = np.min(vmr_one.volume_density_grid).to_value(u.m)
    vmr1_max = np.max(vmr_one.volume_density_grid).to_value(u.m)
    vmr2_min = np.min(vmr_two.volume_density_grid).to_value(u.m)
    vmr2_max = np.max(vmr_two.volume_density_grid).to_value(u.m)

    print(f"Vmr1 min: {vmr1_min}\tVmr2 min: {vmr2_min}")
    print(f"Vmr1 max: {vmr1_max}\tVmr2 max: {vmr2_max}")

    # vdens_comparison_start = np.max(vmr1_min.to_value(u.m), vmr2_min.to_value(u.m))
    # vdens_comparison_stop = np.min(vmr1_max.to_value(u.m), vmr2_max.to_value(u.m))
    comparison_start = max(vmr1_min, vmr2_min)
    comparison_stop = min(vmr1_max, vmr2_max)

    print(f"Comparison extent: {comparison_start} to {comparison_stop}")

    comparison_rs = np.geomspace(start=comparison_start, stop=comparison_stop, num=200)

    vdens_one = vmr_one.volume_density_interpolation(comparison_rs)
    vdens_two = vmr_two.volume_density_interpolation(comparison_rs)

    rprint("[deep_sky_blue3]Volume density comparison:")
    print(vdens_one / vdens_two)

    if (
        vmr_one.column_density_interpolation is None
        or vmr_two.column_density_interpolation is None
    ):
        return
    cdens_one = vmr_one.column_density_interpolation(comparison_rs)
    cdens_two = vmr_two.column_density_interpolation(comparison_rs)

    rprint("[deep_sky_blue3]Column density comparison:")
    print(cdens_one / cdens_two)


def main():
    args = process_args()
    log.debug("Loading input from %s ....", args.parameterfile[0])

    vmc_unxfrmed = pyv.vectorial_model_config_from_yaml(
        pathlib.Path(args.parameterfile[0])
    )
    if vmc_unxfrmed is None:
        print(f"Failed to read {args.parameterfile}!")
        return 1

    r_h = 1.0 * u.AU  # type: ignore
    vmc = apply_input_transform(
        vmc=vmc_unxfrmed, r_h=r_h, xfrm=VmcTransform.cochran_schleicher_93
    )
    vmc_set = [vmc]

    # ec = get_backend_model_selection()

    print("Running python model...")
    py_calc_table = pyv.build_calculation_table(
        vmc_set, extra_config=PythonModelExtraConfig(print_progress=False)
    )
    py_vmr = pyv.unpickle_from_base64(py_calc_table["b64_encoded_vmr"][0])  # type: ignore

    print("Running rust model...")
    rust_calc_table = pyv.build_calculation_table(
        vmc_set,
        extra_config=RustModelExtraConfig(
            rust_input_filename=pathlib.Path("rust_in.yaml"),
            rust_output_filename=pathlib.Path("rust_out.txt"),
        ),
    )
    rust_vmr = pyv.unpickle_from_base64(rust_calc_table["b64_encoded_vmr"][0])  # type: ignore

    print("Running fortran model...")
    fortran_calc_table = pyv.build_calculation_table(
        vmc_set,
        extra_config=FortranModelExtraConfig(
            fortran_input_filename=pathlib.Path("fparam.dat"),
            fortran_output_filename=pathlib.Path("fort.16"),
            r_h=1.0 * u.AU,  # type: ignore
        ),
    )
    fortran_vmr = pyv.unpickle_from_base64(fortran_calc_table["b64_encoded_vmr"][0])  # type: ignore

    print("Python to rust:")
    model_compare(py_vmr, rust_vmr)

    print("Python to fortran:")
    model_compare(py_vmr, fortran_vmr)

    print("Rust to fortran:")
    model_compare(rust_vmr, fortran_vmr)

    # fragment_sputter_plot(vmr=vmr)
    # volume_and_column_density_plots(vmr=vmr)

    # r_kms = vmr.column_density_grid.to_value(u.km)
    # cds = vmr.column_density.to_value(1 / u.cm**2)
    # df = pd.DataFrame({"r_kms": r_kms, "column_density_per_cm2": cds})

    # output_dir = pathlib.Path("output")
    # output_file_stem = pathlib.Path(args.parameterfile[0]).stem
    # output_file_name = pathlib.Path(output_file_stem + ".fits")
    # output_file_path = output_dir / output_file_name
    # remove_file_silent_fail(output_file_path)
    # log.info("Table building complete, writing results to %s ...", output_file_path)
    # out_table.write(output_file_path, format="fits")
    #
    # df.to_csv(output_file_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())

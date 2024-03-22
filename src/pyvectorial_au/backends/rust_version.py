import os
import re
import subprocess
import logging as log
import pathlib
import yaml
import importlib
import tempfile
from itertools import islice
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import astropy.units as u
from pyvectorial_au.post_model_processing.column_density_abel import (
    column_density_from_abel,
)
from pyvectorial_au.post_model_processing.interpolation import (
    interpolate_column_density,
    interpolate_volume_density,
)

from pyvectorial_au.model_input.vectorial_model_config import (
    VectorialModelConfig,
    CometProduction,
    ParentMolecule,
    FragmentMolecule,
    VectorialModelGrid,
)
from pyvectorial_au.model_output.vectorial_model_result import (
    VectorialModelResult,
    FragmentSputterSpherical,
)


"""
    For interfacing with the rust version of the vectorial model
"""


@dataclass
class RustModelExtraConfig:
    bin_path: pathlib.Path = importlib.resources.files(  # type: ignore
        package="pyvectorial_au"
    ) / pathlib.Path("bin/rust_vect")


def run_rust_vectorial_model(
    vmc: VectorialModelConfig, extra_config: RustModelExtraConfig
) -> VectorialModelResult:
    """
    Given path to rust vmodel binary, runs the given model configuration
    """

    # create the temp files and close them to allow the rust executable sole access
    rust_input = tempfile.NamedTemporaryFile(delete=False)
    rust_input.close()
    rust_output = tempfile.NamedTemporaryFile(delete=False)
    rust_output.close()

    rust_input_filename = pathlib.Path(rust_input.name)
    rust_output_filename = pathlib.Path(rust_output.name)

    log.debug(
        "Writing input config file %s to feed rust vectorial model...",
        rust_input_filename,
    )
    write_rust_input_file(vmc, rust_input_filename)

    log.info("Running rust version at %s ...", extra_config.bin_path)
    p1 = subprocess.run(
        args=[
            str(extra_config.bin_path),
            str(rust_input_filename),
            str(rust_output_filename),
        ],
        stdout=open(os.devnull, "wb"),
    )
    log.info("rust vmodel run complete, return code %s", p1.returncode)

    vmr = vmr_from_rust_output(rust_output_filename, vmc)

    # clean up the temp files
    os.unlink(rust_input_filename)
    os.unlink(rust_output_filename)

    interpolate_volume_density(vmr)
    column_density_from_abel(vmr)
    interpolate_column_density(vmr)
    return vmr


# def run_rust_vectorial_model(
#     vmc: VectorialModelConfig, extra_config: RustModelExtraConfig
# ) -> VectorialModelResult:
#     """
#     Given path to rust vmodel binary, runs the given model configuration
#     """
#     rust_input_filename = pathlib.Path(vmc_to_sha256_digest(vmc=vmc)).with_suffix(
#         ".yaml"
#     )
#     rust_output_filename = pathlib.Path(vmc_to_sha256_digest(vmc=vmc)).with_suffix(
#         ".txt"
#     )
#
#     log.debug(
#         "Writing input config file %s to feed rust vectorial model...",
#         rust_input_filename,
#     )
#     write_rust_input_file(vmc, rust_input_filename)
#
#     log.info("Running rust version at %s ...", extra_config.bin_path)
#     p1 = subprocess.run(
#         args=[
#             str(extra_config.bin_path),
#             str(rust_input_filename),
#             str(rust_output_filename),
#         ],
#         stdout=open(os.devnull, "wb"),
#     )
#     log.info("rust vmodel run complete, return code %s", p1.returncode)
#
#     vmr = vmr_from_rust_output(rust_output_filename, vmc)
#
#     interpolate_volume_density(vmr)
#     column_density_from_abel(vmr)
#     interpolate_column_density(vmr)
#     return vmr


def get_rust_header(rust_output_filename: pathlib.Path) -> List[str]:
    with open(rust_output_filename, "r") as f:
        header = list(islice(f, 14))
    return header


def vmc_from_rust_output(rust_output_filename: pathlib.Path) -> VectorialModelConfig:
    header = get_rust_header(rust_output_filename=rust_output_filename)

    base_q = float(re.search(r"Q: ([0-9]\.[0-9]+e[+-]?[0-9]+) mol/s", header[2]).group(1)) / u.s  # type: ignore
    p_tau_d = (
        float(re.search(r"Parent dissociative lifetime: ([0-9]\.[0-9]*e[+-]?[0-9]+) s", header[2]).group(1))  # type: ignore
        * u.s
    )
    p_tau_T = (
        float(re.search(r"Parent total lifetime: ([0-9]\.[0-9]*e[+-]?[0-9]+) s", header[2]).group(1)) * u.s  # type: ignore
    )
    v_outflow = float(
        re.search(r"Parent outflow velocity: ([0-9]\.[0-9]*e[+-]?[0-9]+) m/s", header[2]).group(1)  # type: ignore
    ) * (u.m / u.s)
    sigma = (
        float(re.search(r"Parent cross sectional area: ([0-9]\.[0-9]*e[+-]?[0-9]+) m\^2", header[3]).group(1))  # type: ignore
        * u.m**2  # type: ignore
    )
    f_tau_T = (
        float(re.search(r"Fragment total lifetime: ([0-9]\.[0-9]*e[+-]?[0-9]+) s", header[3]).group(1)) * u.s  # type: ignore
    )
    v_photo = (
        float(re.search(r"Fragment velocity: ([0-9]\.[0-9]*e[+-]?[0-9]+) m/s", header[3]).group(1)) * u.m / u.s  # type: ignore
    )
    radial_points = int(re.search(r"Radial grid size: ([0-9]+)", header[0]).group(1))  # type: ignore
    angular_points = int(re.search(r"Angular grid size: ([0-9]+)", header[0]).group(1))  # type: ignore
    radial_substeps = int(re.search(r"Radial substeps: ([0-9]+)", header[0]).group(1))  # type: ignore
    parent_destruction_level = (
        float(re.search(r"Parent destruction level: ([0-9]+\.[0-9]+)%", header[4]).group(1)) / 100.0  # type: ignore
    )
    fragment_destruction_level = (
        float(re.search(r"Fragment destruction level: ([0-9]+\.[0-9]+)%", header[4]).group(1))  # type: ignore
        / 100.0
    )

    return VectorialModelConfig(
        production=CometProduction(base_q_per_s=base_q.to_value(1 / u.s)),
        parent=ParentMolecule(
            tau_d_s=p_tau_d.to_value(u.s),
            tau_T_s=p_tau_T.to_value(u.s),
            v_outflow_kms=v_outflow.to_value(u.km / u.s),
            sigma_cm_sq=sigma.to_value(u.cm**2),
        ),
        fragment=FragmentMolecule(
            v_photo_kms=v_photo.to_value(u.km / u.s), tau_T_s=f_tau_T.to_value(u.s)
        ),
        grid=VectorialModelGrid(
            radial_points=radial_points,
            angular_points=angular_points,
            radial_substeps=radial_substeps,
            parent_destruction_level=parent_destruction_level,
            fragment_destruction_level=fragment_destruction_level,
        ),
    )


def vmr_from_rust_output(
    rust_output_filename: pathlib.Path, vmc: Optional[VectorialModelConfig]
) -> VectorialModelResult:
    """
    Takes the output of the rust model in rust_output_filename and returns VectorialModelResult
    """

    if vmc is None:
        vmc = vmc_from_rust_output(rust_output_filename=rust_output_filename)

    # volume density starts at line 14 (counting from 1, not from zero), and has vmc.grid.radial_points entries
    volume_density_start_line_number = 13
    volume_density_stop_line_number = (
        volume_density_start_line_number + vmc.grid.radial_points
    )
    volume_density_lines = range(
        volume_density_start_line_number, volume_density_stop_line_number
    )

    # The +1 skips the line of "r, theta, fragment density"
    sputter_starts_at = volume_density_stop_line_number + 1

    volume_density_grid = []
    volume_density = []
    fragment_sputter_list = []

    with open(rust_output_filename) as f:
        header = list(islice(f, 12))
        f.seek(0)
        for i, line in enumerate(f):
            if i in volume_density_lines:
                vals = [float(x) for x in line.split(",")]
                volume_density_grid.extend(vals[0::2])
                volume_density.extend(vals[1::2])
            if i >= sputter_starts_at:
                vals = [float(x) for x in line.split(",")]
                fragment_sputter_list.append(vals)

    # rust outputs are in these units
    volume_density_grid *= u.m  # type: ignore
    volume_density *= 1 / u.m**3  # type: ignore

    fragment_sputter_array = np.array(fragment_sputter_list).astype(float)
    rs = fragment_sputter_array[:, 0] * u.km
    thetas = fragment_sputter_array[:, 1]
    fragment_density = fragment_sputter_array[:, 2] / u.m**3  # type: ignore
    fragment_sputter = FragmentSputterSpherical(
        rs=rs, thetas=thetas, fragment_density=fragment_density
    )

    collision_sphere_radius = (  # type: ignore
        float(re.search("Collision sphere radius: (.*) km", header[6]).group(1)) * u.km  # type: ignore
    )
    coma_radius = float(re.search("Coma radius: (.*) km", header[7]).group(1)) * u.km  # type: ignore
    max_grid_radius = (  # type: ignore
        float(re.search("Max grid extent: (.*) km", header[8]).group(1)) * u.km  # type: ignore
    )

    return VectorialModelResult(
        volume_density_grid=volume_density_grid,
        volume_density=volume_density,
        fragment_sputter=fragment_sputter,
        collision_sphere_radius=collision_sphere_radius,
        max_grid_radius=max_grid_radius,
        coma_radius=coma_radius,
    )


def write_rust_input_file(
    vmc: VectorialModelConfig, rust_input_filename: pathlib.Path
) -> None:
    """
    Takes a valid vectorial model config and produces a valid input file to give to the rust version of the model as input
    Only steady production is currently supported in the rust version
    """

    if vmc.production.time_variation is not None:
        print(
            "Only steady production is supported in the rust version! Running steady production model instead!."
        )

    rust_config_dict = {
        "base_q": float(vmc.production.base_q.to_value(1 / u.s)),  # type: ignore
        "p_tau_d": float(vmc.parent.tau_d.to_value(u.s)),  # type: ignore
        "p_tau_t": float(vmc.parent.tau_T.to_value(u.s)),  # type: ignore
        "v_outflow": float(vmc.parent.v_outflow.to_value(u.m / u.s)),  # type: ignore
        "sigma": float(vmc.parent.sigma.to_value(u.m**2)),  # type: ignore
        "f_tau_t": float(vmc.fragment.tau_T.to_value(u.s)),  # type: ignore
        "v_photo": float(vmc.fragment.v_photo.to_value(u.m / u.s)),  # type: ignore
        "radial_points": int(vmc.grid.radial_points),
        "angular_points": int(vmc.grid.angular_points),
        "radial_substeps": int(vmc.grid.radial_substeps),
        "parent_destruction_level": float(vmc.grid.parent_destruction_level),
        "fragment_destruction_level": float(vmc.grid.fragment_destruction_level),
    }

    with open(rust_input_filename, "w") as out_file:
        yaml.dump(rust_config_dict, out_file, default_flow_style=False)

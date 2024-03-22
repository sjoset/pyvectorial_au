import os
import re
import subprocess
import logging as log
import pathlib
import importlib

import numpy as np
import astropy.units as u
from astropy.units.quantity import Quantity
from itertools import islice
from contextlib import redirect_stdout
from dataclasses import dataclass
from pyvectorial.post_model_processing.column_density_abel import (
    column_density_from_abel,
)
from pyvectorial.post_model_processing.interpolation import (
    interpolate_column_density,
    interpolate_volume_density,
)

from pyvectorial.model_input.vectorial_model_config import VectorialModelConfig
from pyvectorial.model_output.vectorial_model_result import (
    VectorialModelResult,
    FragmentSputterSpherical,
)

"""
    For interfacing with the fortran version of the vectorial model, written by Festou, early 1980s

    Takes a VectorialModelConfig and fills a VectorialModelResult based on its calculations
"""


@dataclass
class FortranModelExtraConfig:
    fortran_input_filename: pathlib.Path
    fortran_output_filename: pathlib.Path
    r_h: Quantity
    read_sputter: bool = True
    bin_path: pathlib.Path = importlib.resources.files(  # type: ignore
        package="pyvectorial"
    ) / pathlib.Path("bin/fvm")


def run_fortran_vectorial_model(
    vmc: VectorialModelConfig, extra_config: FortranModelExtraConfig
) -> VectorialModelResult:
    """
    Given path to fortran binary, runs it by sending it the correct keystrokes
    """

    write_fortran_input_file(vmc, extra_config)

    log.info("Running fortran version at %s ...", extra_config.bin_path)

    # my vm.f consumes 14 enters before the calculation
    enter_key_string = "\n" * 14
    p1 = subprocess.Popen(["echo", enter_key_string], stdout=subprocess.PIPE)
    p2 = subprocess.run(
        f"{extra_config.bin_path}", stdin=p1.stdout, stdout=open(os.devnull, "wb")
    )

    log.info("fortran run complete, return code %s", p2.returncode)

    vmr = vmr_from_fortran_output(
        extra_config.fortran_output_filename, read_sputter=extra_config.read_sputter
    )
    interpolate_volume_density(vmr)
    column_density_from_abel(vmr)
    interpolate_column_density(vmr)
    return vmr


def vmr_from_fortran_output(
    fort16_file: pathlib.Path, read_sputter: bool
) -> VectorialModelResult:
    """
    Takes the output of the fortran code in fort16_file and returns VectorialModelResult
    """

    log.debug("Attempting to extract VectorialModelConfig from %s ... ", fort16_file)

    # Volume density is on line 15 - 27
    fort16_voldens = range(14, 27)
    # Column density is on line 53 - 70
    fort16_coldens = range(52, 70)
    # sputter array starts at line 177 through to the end
    fort16_sputter = 176

    vdg = []
    vd = []
    cdg = []
    cd = []
    sputter = []
    with open(fort16_file) as in_file:
        fort_header = list(islice(in_file, 30))
        in_file.seek(0)
        for i, line in enumerate(in_file):
            if i in fort16_voldens:
                vals = [float(x) for x in line.split()]
                vdg.extend(vals[0::2])
                vd.extend(vals[1::2])
            if i in fort16_coldens:
                vals = [float(x) for x in line.split()]
                cdg.extend(vals[0::2])
                cd.extend(vals[1::2])
            if i >= fort16_sputter and read_sputter:
                vals = [float(x) for x in line.split()]
                sputter.append(vals)

    mgr_line = fort_header[7]
    mgr = float(re.search("DIM=(.*) KM", mgr_line).group(1)) * u.km  # type: ignore
    cs_line = fort_header[10]
    csphere = float(re.search("is: (.*) cm", cs_line).group(1)) * u.cm  # type: ignore
    cr_line = fort_header[7]
    cr = float(re.search(r"RCOMA=(.*)\(KM\)", cr_line).group(1)) * u.km  # type: ignore

    # fortran outputs are in these units
    vdg *= u.km  # type: ignore
    cdg *= u.km  # type: ignore
    vd *= 1 / u.cm**3  # type: ignore
    cd *= 1 / u.cm**2  # type: ignore

    sputter = np.array(sputter).astype(float)
    rs = sputter[:, 0] * u.km
    thetas = sputter[:, 1]
    fragment_density = sputter[:, 2] / u.cm**3  # type: ignore
    fs = FragmentSputterSpherical(
        rs=rs, thetas=thetas, fragment_density=fragment_density
    )

    log.debug("VectorialModelResult extraction complete.")

    return VectorialModelResult(
        volume_density_grid=vdg,
        volume_density=vd,
        fragment_sputter=fs,
        collision_sphere_radius=csphere,
        max_grid_radius=mgr,
        coma_radius=cr,
        column_density_grid=cdg,
        column_density=cd,
    )


def fragment_theory_count_from_fortran_output(
    fortran_output_filename: pathlib.Path,
) -> float:
    with open(fortran_output_filename) as in_file:
        fort_header = list(islice(in_file, 30))
    nft_line = fort_header[29]
    nft = float(re.search("IS  (.*)  TOTAL", nft_line).group(1))  # type: ignore

    return nft


def fragment_grid_count_from_fortran_output(
    fortran_output_filename: pathlib.Path,
) -> float:
    with open(fortran_output_filename) as in_file:
        fort_header = list(islice(in_file, 30))
    nfg_line = fort_header[28]
    nfg = float(re.search("COMA: (.*)$", nfg_line).group(1))  # type: ignore

    return nfg


def write_fortran_input_file(
    vmc: VectorialModelConfig, ec: FortranModelExtraConfig
) -> None:
    """
    Takes a VectorialModelConfig and produces a fortran input file,
    as long as the production is steady
    """

    if vmc.production.time_variation is not None:
        print(
            "Only steady production is currently supported for producing fortran input files! Running steady production model instead!"
        )

    log.debug(
        "Writing input config file %s to feed fortran vectorial model...",
        ec.fortran_input_filename,
    )

    # default values that do not affect volume or column density, only aperture brightness, which we don't use
    # small delta stos the IUE aperture sizes hard-coded into the fortran version from being too large
    delta = 0.01
    g_factor = 2.33e-4
    comet_name = "Default comet"
    fragment_name = "Default fragment"

    # TODO: binned time production should also be handled here, but fortran only supports 20 bins of time division
    #   so we would have to check if the input had 20 or less time bins first
    # TODO: parent and fragment destruction levels are hard-coded to the defaults of the python model in sbpy so that
    #   the calculations will match

    with open(ec.fortran_input_filename, "w") as out_file:
        with redirect_stdout(out_file):
            print(f"{comet_name}")
            print(f"{ec.r_h.to(u.AU).value}  {delta}")
            # length of production array: only base production rate for 120 days
            print("1")
            print(f"{vmc.production.base_q.to_value(1/u.s)}  120.0")
            # fill in dummy values for the rest of the array
            for _ in range(19):
                print("0.0 121")
            # parent info - speed, total & photo lifetime, destruction level
            print(f"{vmc.parent.v_outflow.to_value(u.km/u.s)}")
            print(f"{vmc.parent.tau_T.to_value(u.s)}")
            print(f"{vmc.parent.tau_d.to_value(u.s)}")
            print(f"{vmc.grid.parent_destruction_level*100}")
            # fragment info - gfactor, speed, total lifetime, destruction level
            print(f"{fragment_name}")
            print(f"{g_factor}")
            print(f"{vmc.fragment.v_photo.to_value(u.km/u.s)}")
            print(f"{vmc.fragment.tau_T.to_value(u.s)}")
            print(f"{vmc.grid.fragment_destruction_level*100}")
            # Custom aperture size, unused for our purposes so these are dummy values
            print("  1.3       3.6 ")
            # print("  0.000001       0.00001")

    log.debug("Writing fortran input file complete.")

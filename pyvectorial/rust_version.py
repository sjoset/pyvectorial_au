import os
import re
import subprocess
import logging as log
import pathlib
import yaml

import numpy as np
import astropy.units as u
from itertools import islice

from .vmconfig import VectorialModelConfig
from .vmresult import VectorialModelResult, FragmentSputterPolar

"""
    For interfacing with the rust version of the vectorial model

    Takes a VectorialModelConfig and fills a VectorialModelResult based on its calculations
"""


def run_rust_vmodel(
    vmc: VectorialModelConfig, rust_vmodel_bin_path: pathlib.Path
) -> VectorialModelResult:
    """
    Given path to rust vmodel binary, runs it by sending it the correct keystrokes
    """

    rust_input_filename = pathlib.Path("rust_input.yaml")
    rust_output_filename = pathlib.Path("rust_output.txt")

    log.debug("Writing file %s to feed rust vectorial model...", rust_input_filename)
    _write_rust_input_file(vmc, rust_input_filename)

    log.info("Running rust version at %s ...", rust_vmodel_bin_path)
    p1 = subprocess.run(
        f"{rust_vmodel_bin_path} {rust_input_filename} {rust_output_filename}",
        stdout=open(os.devnull, "wb"),
    )
    log.info("rust vmodel run complete, return code %s", p1.returncode)

    return vmr_from_rust_output(rust_output_filename)


def vmr_from_rust_output(
    rust_output_filename: pathlib.Path, read_sputter: bool = True
) -> VectorialModelResult:
    """
    Takes the output of the rust code in rust_output_filename and returns VectorialModelResult
    """

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
    with open(rust_output_filename) as in_file:
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
    mgr = float(re.search("DIM=(.*) KM", mgr_line).group(1)) * u.km
    cs_line = fort_header[10]
    csphere = float(re.search("is: (.*) cm", cs_line).group(1)) * u.cm
    nft_line = fort_header[29]
    nft = float(re.search("IS  (.*)  TOTAL", nft_line).group(1))
    nfg_line = fort_header[28]
    nfg = float(re.search("COMA: (.*)$", nfg_line).group(1))
    cr_line = fort_header[7]
    cr = float(re.search(r"RCOMA=(.*)\(KM\)", cr_line).group(1)) * u.km

    # rust outputs are in these units
    vdg *= u.km
    cdg *= u.km
    vd *= 1 / u.cm**3
    cd *= 1 / u.cm**2

    # print(sputter)
    sputter = np.array(sputter).astype(float)
    rs = sputter[:, 0] * u.km
    thetas = sputter[:, 1]
    fragment_density = sputter[:, 2] / u.cm**3
    fs = FragmentSputterPolar(rs=rs, thetas=thetas, fragment_density=fragment_density)

    return VectorialModelResult(
        volume_density_grid=vdg,
        volume_density=vd,
        column_density_grid=cdg,
        column_density=cd,
        fragment_sputter=fs,
        volume_density_interpolation=None,
        column_density_interpolation=None,
        collision_sphere_radius=csphere,
        max_grid_radius=mgr,
        coma_radius=cr,
        num_fragments_theory=nft,
        num_fragments_grid=nfg,
    )


def _write_rust_input_file(
    vmc: VectorialModelConfig, rust_input_filename: pathlib.Path
) -> None:
    """
    Takes a valid python vectorial model config and produces a valid rust vmodel input file
    Only steady production is currently supported in the rust version
    """

    if vmc.production.time_variation_type is not None:
        log.info("Only steady production is supported in the rust version! Skipping.")
        return

    rust_config_dict = {
        "base_q": vmc.production.base_q.to_value(1 / u.s),
        "p_tau_d": vmc.parent.tau_d.to_value(u.s),
        "p_tau_t": vmc.parent.tau_T.to_value(u.s),
        "v_outflow": vmc.parent.v_outflow.to_value(u.m / u.s),
        "sigma": vmc.parent.sigma.to_value(u.m**2),
        "f_tau_t": vmc.fragment.tau_T.to_value(u.s),
        "v_photo": vmc.fragment.v_photo.to_value(u.s),
        "radial_points": int(vmc.grid.radial_points),
        "angular_points": int(vmc.grid.angular_points),
        "radial_substeps": int(vmc.grid.radial_substeps),
        "parent_destruction_level": 0.99,
        "fragment_destruction_level": 0.95,
    }

    with open(rust_input_filename, "w") as out_file:
        yaml.dump(rust_config_dict, out_file, default_flow_style=False)

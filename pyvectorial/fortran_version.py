
import os
import re
import subprocess
import logging as log

import numpy as np
import astropy.units as u
import logging as log
from itertools import islice
from contextlib import redirect_stdout

from .vmconfig import VectorialModelConfig
from .vmresult import VectorialModelResult, FragmentSputterPolar

"""
    For interfacing with the fortran version of the vectorial model, written by Festou, early 1980s
"""


def run_fortran_vmodel(vmc: VectorialModelConfig) -> None:

    """
        Given path to fortran binary, runs it by sending it the correct keystrokes
    """

    log.debug("Writing file to feed fortran vectorial model...")
    _produce_fortran_fparam(vmc)

    fortran_vmodel_binary = vmc.etc['vmodel_binary']
    log.info("Running fortran version at %s ...", fortran_vmodel_binary)

    # my vm.f consumes 14 enters before the calculation
    enter_key_string = "\n" * 14
    p1 = subprocess.Popen(["echo", enter_key_string], stdout=subprocess.PIPE)
    p2 = subprocess.run(f"{fortran_vmodel_binary}", stdin=p1.stdout, stdout=open(os.devnull, 'wb'))

    log.info("fortran run complete, return code %s", p2.returncode)


def get_result_from_fortran(fort16_file: str, read_sputter: bool = True) -> VectorialModelResult:

    """
        Takes the output of the fortran code in fort16_file and returns VectorialModelResult
    """

    # Volume density is on line 15 - 27
    fort16_voldens = range(14, 27)
    # Column density is on line 53 - 70
    fort16_coldens = range(52, 70)
    # sputter array starts at line 175 through to the end
    fort16_sputter = 175

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
    mgr = float(re.search('DIM=(.*) KM', mgr_line).group(1)) * u.km
    cs_line = fort_header[10]
    csphere = float(re.search('is: (.*) cm', cs_line).group(1)) * u.cm
    nft_line = fort_header[29]
    nft = float(re.search('IS  (.*)  TOTAL', nft_line).group(1))
    nfg_line = fort_header[28]
    nfg = float(re.search('COMA: (.*)$', nfg_line).group(1))
    cr_line = fort_header[7]
    cr = float(re.search('RCOMA=(.*)\(KM\)', cr_line).group(1)) * u.km

    # fortran outputs are in these units
    vdg *= u.km
    cdg *= u.km
    vd *= 1/u.cm**3
    cd *= 1/u.cm**2

    sputter = np.array(sputter).astype(float)
    # sputter = sputter.astype(float)
    rs = sputter[:, 0] * u.km
    thetas = sputter[:, 1]
    fragment_density = sputter[:, 2] / u.cm**3
    fs = FragmentSputterPolar(rs=rs, thetas=thetas, fragment_density=fragment_density)

    return VectorialModelResult(
            volume_density_grid=vdg, volume_density=vd,
            column_density_grid=cdg, column_density=cd,
            fragment_sputter=fs,
            volume_density_interpolation=None, column_density_interpolation=None,
            collision_sphere_radius=csphere, max_grid_radius=mgr, coma_radius=cr,
            num_fragments_theory=nft, num_fragments_grid=nfg
            )


def _produce_fortran_fparam(vmc: VectorialModelConfig) -> None:

    """
        Takes a valid python config and produces a valid fortran input file
        as long as the production is steady
    """

    if vmc.production.time_variation_type is not None:
        log.info("Only steady production is supported for producing fortran input files! Skipping.")
        return

    # TODO: binned time production should also be handled here, but fortran only supports 20 bins of time division
    #   so we would have to check if the input had 20 or less time bins first
    # TODO: parent and fragment destruction levels are hard-coded to the defaults in the python version
    fparam_outfile = vmc.etc['in_file']

    with open(fparam_outfile, 'w') as out_file:
        with redirect_stdout(out_file):
            print(f"{vmc.comet.name}")
            print(f"{vmc.comet.rh.to(u.AU).value}  {vmc.etc['delta']}")
            # length of production array: only base production rate for 60 days
            print("1")
            print(f"{vmc.production.base_q.to(1/u.s).value}  60.0")
            # fill in dummy values for the rest of the array
            for _ in range(19):
                print("0.0 61")
            # parent info - speed, total & photo lifetime, destruction level
            print(f"{vmc.parent.v_outflow.to(u.km/u.s).value}")
            print(f"{vmc.parent.tau_T.to(u.s).value}")
            print(f"{vmc.parent.tau_d.to(u.s).value}")
            print("99.0")
            # fragment info - gfactor, speed, total lifetime, destruction level
            print(f"{vmc.fragment.name}")
            print(f"{vmc.etc['g_factor']}")
            print(f"{vmc.fragment.v_photo.to(u.km/u.s).value}")
            print(f"{vmc.fragment.tau_T.to(u.s).value}")
            print("95.0")
            # Custom aperture size, unused for our purposes so these are dummy values
            print("  100.000000       100.00000")

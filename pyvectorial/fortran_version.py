
import os
import subprocess
import logging as log

import numpy as np
import astropy.units as u
import logging as log
from contextlib import redirect_stdout
from .vmconfig import VectorialModelConfig

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


def read_fortran_vm_output(fort16_file, read_sputter=False):

    """
        Takes the output of the fortran code and returns the volume & column density,
        along the gridpoints of each
    """

    # Volume density is on line 15 - 27
    fort16_voldens = range(14, 27)
    # Column density is on line 53 - 70
    fort16_coldens = range(52, 70)
    # sputter array starts at line 175 through to the end
    fort16_sputter = 175

    vol_grid_points = []
    col_grid_points = []
    col_dens = []
    vol_dens = []
    sputter = []
    with open(fort16_file) as in_file:
        for i, line in enumerate(in_file):
            if i in fort16_voldens:
                vals = [float(x) for x in line.split()]
                vol_grid_points.extend(vals[0::2])
                vol_dens.extend(vals[1::2])
            if i in fort16_coldens:
                vals = [float(x) for x in line.split()]
                col_grid_points.extend(vals[0::2])
                col_dens.extend(vals[1::2])
            if i >= fort16_sputter and read_sputter:
                vals = [float(x) for x in line.split()]
                sputter.append(vals)

    # These are the units the fortran output uses
    vol_grid_points *= u.km
    col_grid_points *= u.km
    vol_dens *= 1/u.cm**3
    col_dens *= 1/u.cm**2

    sputter = np.array(sputter)
    sputter = sputter.astype(float)

    return vol_grid_points, vol_dens, col_grid_points, col_dens, sputter


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



import time

import logging as log
import astropy.units as u
import importlib.metadata as impm
import pyvectorial as pyv

from astropy.table import QTable
from multiprocessing import Pool
from typing import List, Tuple


def run_vmodel_timed(vmc: pyv.VectorialModelConfig) -> Tuple:
    """
    Service function that takes a vmc, runs a model, and returns results + timing information.

    Returns the encoded coma (using the dill library) because python multiprocessing wants
    to pickle return values to send them back to the main calling process.  The coma can't be
    pickled by the stock python pickler so we pickle it here with dill and things are fine
    """

    t_i = time.time()
    coma_pickled = pyv.pickle_to_base64(pyv.run_vmodel(vmc))
    t_f = time.time()

    return (coma_pickled, (t_f - t_i) * u.s)


def build_calculation_table(
    vmc_set: List[pyv.VectorialModelConfig], parallelism: int = 1
) -> QTable:
    """
    Take a set of model configs, run them, and return QTable with results of input vmc,
    resulting comae, and model run time
    Uses the multiprocessing module to parallelize the model running, with the number of
    concurrent processes passed in as 'parallelism'
    """

    sbpy_ver = impm.version("sbpy")
    calculation_table = QTable(
        names=("b64_encoded_vmc", "vmc_hash", "b64_encoded_coma"),
        dtype=("U", "U", "U"),
        meta={"sbpy_ver": sbpy_ver},
    )

    t_i = time.time()
    log.info(
        "Running calculation of %s models with pool size of %s ...",
        len(vmc_set),
        parallelism,
    )
    with Pool(parallelism) as vm_pool:
        model_results = vm_pool.map(run_vmodel_timed, vmc_set)
    t_f = time.time()
    log.info("Pooled calculations complete, time: %s", (t_f - t_i) * u.s)

    times_list = []
    for i, vmc in enumerate(vmc_set):
        pickled_coma = model_results[i][0]
        times_list.append(model_results[i][1])
        pickled_vmc = pyv.pickle_to_base64(vmc)

        calculation_table.add_row((pickled_vmc, pyv.hash_vmc(vmc), pickled_coma))

    calculation_table.add_column(times_list, name="model_run_time")

    # now that the model runs are finished, add the config info as columns to the table
    add_vmc_columns(calculation_table)

    return calculation_table


def add_vmc_columns(qt: QTable) -> None:
    """
    Take a QTable of finished vectorial model calculations and add information
    from the VectorialModelConfig as columns in the given table
    """
    vmc_list = [pyv.unpickle_from_base64(row["b64_encoded_vmc"]) for row in qt]  # type: ignore

    qt.add_column([vmc.production.base_q for vmc in vmc_list], name="base_q")

    qt.add_column([vmc.parent.name for vmc in vmc_list], name="parent_molecule")
    qt.add_column([vmc.parent.tau_d for vmc in vmc_list], name="parent_tau_d")
    qt.add_column([vmc.parent.tau_T for vmc in vmc_list], name="parent_tau_T")
    qt.add_column([vmc.parent.sigma for vmc in vmc_list], name="parent_sigma")
    qt.add_column([vmc.parent.v_outflow for vmc in vmc_list], name="v_outflow")

    qt.add_column([vmc.fragment.name for vmc in vmc_list], name="fragment_molecule")
    qt.add_column([vmc.fragment.tau_T for vmc in vmc_list], name="fragment_tau_T")
    qt.add_column([vmc.fragment.v_photo for vmc in vmc_list], name="v_photo")

    qt.add_column([vmc.comet.rh for vmc in vmc_list], name="r_h")

    qt.add_column([vmc.grid.radial_points for vmc in vmc_list], name="radial_points")
    qt.add_column([vmc.grid.angular_points for vmc in vmc_list], name="angular_points")
    qt.add_column(
        [vmc.grid.radial_substeps for vmc in vmc_list], name="radial_substeps"
    )

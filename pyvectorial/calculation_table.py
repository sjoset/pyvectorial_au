import time

import logging as log
import astropy.units as u
import importlib.metadata as impm

from astropy.table import QTable
from astropy.units.quantity import Quantity
from multiprocessing import Pool
from typing import List, Tuple, Union
from functools import partial

from .vectorial_model_config import VectorialModelConfig
from .vectorial_model_result import VectorialModelResult
from .vectorial_model_runner import run_vectorial_model
from .python_version import PythonModelExtraConfig
from .fortran_version import FortranModelExtraConfig
from .rust_version import RustModelExtraConfig
from .pickle_encoding import pickle_to_base64, unpickle_from_base64
from .hashing import hash_vmc


def run_vmodel_timed(
    vmc: VectorialModelConfig,
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ],
) -> Tuple[str, Quantity]:
    """
    Service function that takes a vmc, runs a model, and returns results + timing information.

    Returns the encoded coma (using the dill library) because python multiprocessing wants
    to pickle return values to send them back to the main calling process.  The coma can't be
    pickled by the stock python pickler so we pickle it here with dill and things are fine
    """

    t_i = time.time()
    vmr_pickled = pickle_to_base64(run_vectorial_model(vmc, extra_config))
    t_f = time.time()

    return (vmr_pickled, (t_f - t_i) * u.s)


def build_calculation_table(
    vmc_set: List[VectorialModelConfig],
    parallelism: int = 1,
    extra_config=PythonModelExtraConfig(print_progress=False),
) -> QTable:
    """
    Take a set of model configs, run them, and return QTable with results of input vmc,
    resulting comae, and model run time
    Uses the multiprocessing module to parallelize the model running, with the number of
    concurrent processes passed in as 'parallelism'
    """

    sbpy_ver = impm.version("sbpy")
    calculation_table = QTable(
        names=("vmc_hash", "b64_encoded_vmc", "b64_encoded_vmr"),
        dtype=("U", "U", "U"),
        meta={"sbpy_ver": sbpy_ver},
    )

    t_i = time.time()
    log.info(
        "Running calculation of %s models with pool size of %s ...",
        len(vmc_set),
        parallelism,
    )

    run_vmodel_timed_mappable_func = partial(
        run_vmodel_timed, extra_config=extra_config
    )
    with Pool(parallelism) as vm_pool:
        model_results = vm_pool.map(run_vmodel_timed_mappable_func, vmc_set)
    t_f = time.time()
    log.info("Pooled calculations complete, time: %s", (t_f - t_i) * u.s)

    times_list = []
    for i, vmc in enumerate(vmc_set):
        pickled_vmr = model_results[i][0]
        times_list.append(model_results[i][1])
        pickled_vmc = pickle_to_base64(vmc)

        calculation_table.add_row((hash_vmc(vmc), pickled_vmc, pickled_vmr))

    calculation_table.add_column(times_list, name="model_run_time")

    # now that the model runs are finished, add the config info as columns to the table
    add_vmc_columns(calculation_table)

    return calculation_table


def add_vmc_columns(qt: QTable) -> None:
    """
    Take a QTable of finished vectorial model calculations and add information
    from the VectorialModelConfig as columns in the given table
    """
    vmc_list = [unpickle_from_base64(row["b64_encoded_vmc"]) for row in qt]  # type: ignore

    qt.add_column([vmc.production.base_q for vmc in vmc_list], name="base_q")

    qt.add_column([vmc.parent.tau_d for vmc in vmc_list], name="parent_tau_d")
    qt.add_column([vmc.parent.tau_T for vmc in vmc_list], name="parent_tau_T")
    qt.add_column([vmc.parent.sigma for vmc in vmc_list], name="parent_sigma")
    qt.add_column([vmc.parent.v_outflow for vmc in vmc_list], name="v_outflow")

    qt.add_column([vmc.fragment.tau_T for vmc in vmc_list], name="fragment_tau_T")
    qt.add_column([vmc.fragment.v_photo for vmc in vmc_list], name="v_photo")

    qt.add_column([vmc.grid.radial_points for vmc in vmc_list], name="radial_points")
    qt.add_column([vmc.grid.angular_points for vmc in vmc_list], name="angular_points")
    qt.add_column(
        [vmc.grid.radial_substeps for vmc in vmc_list], name="radial_substeps"
    )
    qt.add_column(
        [vmc.grid.parent_destruction_level for vmc in vmc_list],
        name="parent_destruction_level",
    )
    qt.add_column(
        [vmc.grid.fragment_destruction_level for vmc in vmc_list],
        name="fragment_destruction_level",
    )

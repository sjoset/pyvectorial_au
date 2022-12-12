
import time
import importlib.metadata as impm
import astropy.units as u

from astropy.table import QTable
from typing import List
from multiprocessing import Pool

from .vmconfig import VectorialModelConfig
from .vmrunner import run_vmodel
from .hashing import hash_vmc, pickle_to_base64


"""
    Functionality for taking a set of VectorialModelConfigs, running sbpy VectorialModels with them,
    and returning an astropy QTable with the results
"""


# Service function that takes a vmc, runs a model, and returns results + timing information
# We map this function over a set of VectorialModelConfigs to get our list of completed models
def run_vmodel_timed(vmc: VectorialModelConfig):

    """
        Return the encoded coma (using the dill library) because python multiprocessing wants
        to pickle return values to send them back to the main calling process.  The coma can't be
        pickled by the stock python pickler so we pickle it here with dill and things are fine
    """

    t_i = time.time()
    coma_pickled = pickle_to_base64(run_vmodel(vmc))
    t_f = time.time()

    return (coma_pickled, (t_f - t_i)*u.s)


# TODO: add table reading and expanding of vmc from table_read_test to library
def build_calculation_table(vmc_set: List[VectorialModelConfig], parallelism=1) -> QTable:
    """
        Take a set of model configs, run them, and return QTable with results of input vmc,
        resulting comae, and model run time
        Uses the multiprocessing module to parallelize the model running, with the number of
        concurrent processes passed in as 'parallelism'
    """

    sbpy_ver = impm.version("sbpy")
    calculation_table = QTable(names=('b64_encoded_vmc', 'vmc_hash', 'b64_encoded_coma'), dtype=('U', 'U', 'U'), meta={'sbpy_ver': sbpy_ver})

    t_i = time.time()
    print(f"Running calculation of {len(vmc_set)} models with pool size of {parallelism} ...")
    with Pool(parallelism) as vm_pool:
        model_results = vm_pool.map(run_vmodel_timed, vmc_set)
    t_f = time.time()
    print(f"Pooled calculations complete, time: {(t_f - t_i)*u.s}")

    times_list = []
    for i, vmc in enumerate(vmc_set):

        pickled_coma = model_results[i][0]
        times_list.append(model_results[i][1])
        pickled_vmc = pickle_to_base64(vmc)

        calculation_table.add_row((pickled_vmc, hash_vmc(vmc), pickled_coma))

    calculation_table.add_column(times_list, name='model_run_time')

    return calculation_table

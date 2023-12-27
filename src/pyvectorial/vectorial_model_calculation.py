from dataclasses import dataclass
import time
import importlib.metadata as impm
from typing import List, Union
from functools import partial
from multiprocessing import Pool

import astropy.units as u

from pyvectorial.backends.fortran_version import FortranModelExtraConfig
from pyvectorial.backends.python_version import PythonModelExtraConfig
from pyvectorial.backends.rust_version import RustModelExtraConfig
from pyvectorial.encoding_and_hashing import unpickle_from_base64
from pyvectorial.vectorial_model_config import VectorialModelConfig
from pyvectorial.vectorial_model_result import VectorialModelResult
from pyvectorial.vectorial_model_runner import run_vectorial_model_timed


@dataclass
class VMCalculation:
    vmc: VectorialModelConfig
    vmr: VectorialModelResult
    execution_time_s: float
    vectorial_model_backend: str
    vectorial_model_version: str


# TODO: fortran version should not be allowed to be run in parallel - the input and output filenames
# are fixed, so any parallel i/o gets overwritten by multiple processes and turns into garbage
def run_vectorial_models_pooled(
    vmc_set: List[VectorialModelConfig],
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ] = PythonModelExtraConfig(print_progress=False),
    parallelism: int = 1,
) -> List[VMCalculation]:
    pool_start_time = time.time()

    run_vmodel_timed_mappable_func = partial(
        run_vectorial_model_timed, extra_config=extra_config
    )
    with Pool(parallelism) as vm_pool:
        model_results = vm_pool.map(run_vmodel_timed_mappable_func, vmc_set)
    pool_end_time = time.time()
    print(f"Total run time: {pool_end_time - pool_start_time} seconds")

    vmrs = [x[0] for x in model_results]
    execution_times = [x[1] for x in model_results]

    # TODO: move this to the three backend files: config -> str
    if isinstance(extra_config, PythonModelExtraConfig):
        vectorial_model_backend = "python"
        vectorial_model_version = impm.version("sbpy")
    elif isinstance(extra_config, FortranModelExtraConfig):
        vectorial_model_backend = "fortran"
        vectorial_model_version = "1.0.0"
    elif isinstance(extra_config, RustModelExtraConfig):
        vectorial_model_backend = "rust"
        vectorial_model_version = "0.1.0"

    vm_calculation_list = [
        VMCalculation(
            vmc=vmc,
            vmr=unpickle_from_base64(vmr),
            execution_time_s=t.to_value(u.s),  # type: ignore
            vectorial_model_backend=vectorial_model_backend,
            vectorial_model_version=vectorial_model_version,
        )
        for (vmc, vmr, t) in zip(vmc_set, vmrs, execution_times)
    ]

    return vm_calculation_list

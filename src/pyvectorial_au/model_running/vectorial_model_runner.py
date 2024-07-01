from functools import partial
from multiprocessing import Pool
import time
import pathlib
import importlib.metadata as impm
from typing import Union, Optional, List

from pyvectorial_au.db.cache_io import get_saved_model_from_cache, save_model_to_cache
from pyvectorial_au.model_input.vectorial_model_config import VectorialModelConfig
from pyvectorial_au.backends.python_version import (
    PythonModelExtraConfig,
    run_python_vectorial_model,
)
from pyvectorial_au.backends.fortran_version import (
    FortranModelExtraConfig,
    run_fortran_vectorial_model,
)
from pyvectorial_au.backends.rust_version import (
    RustModelExtraConfig,
    run_rust_vectorial_model,
)
from pyvectorial_au.encoding.encoding_and_hashing import (
    pickle_to_base64,
    vmc_to_sha256_digest,
)
from pyvectorial_au.model_output.vectorial_model_calculation import (
    EncodedVMCalculation,
    VMCalculation,
)


def verbose_print(x, verbose: bool) -> None:
    if verbose:
        print(x)


def run_vectorial_model_single(
    vmc: VectorialModelConfig,
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ] = PythonModelExtraConfig(print_progress=False),
    vectorial_model_cache_path: Optional[pathlib.Path] = None,
) -> EncodedVMCalculation:
    """
    Given a vectorial model config and database path, look up the result in the database and return
    the already-run model.  If it is not in the database, run it and add it to the db.
    If no db path is specified, just run the model and return the result.
    This was meant to run in its own process through multiprocessing so it must
    create its own database engine.
    """
    # attempt to load model if we find it in the database
    if vectorial_model_cache_path:
        evmc = get_saved_model_from_cache(
            vmc=vmc, vectorial_model_cache_path=vectorial_model_cache_path
        )
        if evmc:
            return evmc

    model_start_time = time.time()
    model_function = None

    # TODO: rust version is hard-coded here - look for a better way to do this - maybe use the bin path to ask the rust executable with 'rust_vectorial_model --version' - need to add this functionality to rust version
    if isinstance(extra_config, PythonModelExtraConfig):
        vectorial_model_backend = "python"
        vectorial_model_version = impm.version("sbpy")
        model_function = run_python_vectorial_model
    elif isinstance(extra_config, FortranModelExtraConfig):
        vectorial_model_backend = "fortran"
        vectorial_model_version = "1.0.0"
        model_function = run_fortran_vectorial_model
    elif isinstance(extra_config, RustModelExtraConfig):
        vectorial_model_backend = "rust"
        vectorial_model_version = "0.1.0"
        model_function = run_rust_vectorial_model

    vmr = model_function(vmc=vmc, extra_config=extra_config)  # type: ignore
    execution_time_s = time.time() - model_start_time

    evmc = EncodedVMCalculation(
        vmc=vmc,
        evmr=pickle_to_base64(vmr),
        execution_time_s=execution_time_s,
        vectorial_model_backend=vectorial_model_backend,
        vectorial_model_version=vectorial_model_version,
    )

    # save model into db cache
    if vectorial_model_cache_path:
        save_model_to_cache(
            evmc=evmc, vectorial_model_cache_path=vectorial_model_cache_path
        )

    return evmc


def run_vectorial_models(
    vmc_list: List[VectorialModelConfig],
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ] = PythonModelExtraConfig(print_progress=False),
    parallelism: int = 1,
    vectorial_model_cache_path: Optional[pathlib.Path] = None,
    verbose: bool = False,
) -> List[VMCalculation]:
    """
    Take a list of model configs and run them in parallel, {parallelism} at a time.

    If a cache path is specified, then we only run unique models to avoid trying to
    save the same model to the db at the same time
    """
    if vectorial_model_cache_path is not None:
        # if we are caching the models, run the unique model configs and cache them,
        # then build the results list like normal
        run_unique_models_only(
            vmc_list=vmc_list,
            extra_config=extra_config,
            parallelism=parallelism,
            vectorial_model_cache_path=vectorial_model_cache_path,
            verbose=verbose,
        )

    verbose_print("Caching done, running...", verbose=verbose)

    # The fortran version uses fixed file names for input and output, so running multiple in parallel
    # would clobber each other's input and output files
    if isinstance(extra_config, FortranModelExtraConfig):
        verbose_print("Forcing no parallelism for fortran version!", verbose=verbose)
        parallelism = 1

    pool_start_time = time.time()

    run_vmodel_timed_mappable_func = partial(
        run_vectorial_model_single,
        extra_config=extra_config,
        vectorial_model_cache_path=vectorial_model_cache_path,
    )

    with Pool(parallelism) as vm_pool:
        vmcalc_list = vm_pool.map(run_vmodel_timed_mappable_func, vmc_list)

    pool_end_time = time.time()
    verbose_print(
        f"Total results assembly time: {pool_end_time - pool_start_time} seconds",
        verbose=verbose,
    )

    return [VMCalculation.from_encoded(x) for x in vmcalc_list]


def run_unique_models_only(
    vmc_list: List[VectorialModelConfig],
    vectorial_model_cache_path: pathlib.Path,
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ] = PythonModelExtraConfig(print_progress=False),
    parallelism: int = 1,
    verbose: bool = False,
) -> None:
    """
    This compares the given vmc_list hashes and runs only the models which are different,
    saving the results into the database at vectorial_model_cache_path - no results are returned directly.
    This allows us to later "run" the full vmc_list and pull the results out of the database
    in the same order.
    """
    # The fortran version uses fixed file names for input and output, so running multiple in parallel
    # would clobber each other's input and output files without some extra temporary-directory work
    # that isn't really worth it
    if isinstance(extra_config, FortranModelExtraConfig):
        verbose_print("Forcing no parallelism for fortran version!", verbose=verbose)
        parallelism = 1

    vmc_hashes = [vmc_to_sha256_digest(x) for x in vmc_list]
    unique_vmc_hashes = set(vmc_hashes)
    reduced_vmc_set = [
        vmc_list[vmc_hashes.index(unique_hash)] for unique_hash in unique_vmc_hashes
    ]
    verbose_print(
        f"Number of unique models to be run and cached: {len(reduced_vmc_set)}",
        verbose=verbose,
    )

    pool_start_time = time.time()

    run_vmodel_timed_mappable_func = partial(
        run_vectorial_model_single,
        extra_config=extra_config,
        vectorial_model_cache_path=vectorial_model_cache_path,
    )

    with Pool(parallelism) as vm_pool:
        # run the results and let them be saved in db
        _ = vm_pool.map(run_vmodel_timed_mappable_func, reduced_vmc_set)

    pool_end_time = time.time()
    verbose_print(
        f"Total run time on unique models: {pool_end_time - pool_start_time} seconds",
        verbose=verbose,
    )

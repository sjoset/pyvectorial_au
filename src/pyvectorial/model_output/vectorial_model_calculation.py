from dataclasses import dataclass
import pathlib
import time
import importlib.metadata as impm
from typing import List, Optional, Union, TypeAlias
from functools import partial
from multiprocessing import Pool

import pandas as pd
import dill
from pydantic import TypeAdapter

from pyvectorial.backends.fortran_version import (
    FortranModelExtraConfig,
    run_fortran_vectorial_model,
)
from pyvectorial.backends.python_version import (
    PythonModelExtraConfig,
    run_python_vectorial_model,
)
from pyvectorial.backends.rust_version import (
    RustModelExtraConfig,
    run_rust_vectorial_model,
)
from pyvectorial.db.vectorial_model_cache import (
    VMCached,
    get_vm_cache_db_session,
    initialize_vectorial_model_cache,
)
from pyvectorial.encoding.encoding_and_hashing import (
    compress_vmr_string,
    decompress_vmr_string,
    pickle_to_base64,
    unpickle_from_base64,
    vmc_to_sha256_digest,
)
from pyvectorial.model_input.vectorial_model_config import VectorialModelConfig
from pyvectorial.model_output.vectorial_model_result import VectorialModelResult

EncodedVectorialModelResult: TypeAlias = str


@dataclass
class EncodedVMCalculation:
    """
    Uses a pickled VectorialModelResult (using the dill library) because python multiprocessing wants
    to pickle return values to send them back to the main calling process.  The coma can't be
    pickled by the stock python pickler so we have to encode it ourselves with dill. This allows us
    to return this data structure from a job started in parallel, where returning VMCalculation directly
    would fail.
    """

    vmc: VectorialModelConfig
    evmr: EncodedVectorialModelResult
    execution_time_s: float
    vectorial_model_backend: str
    vectorial_model_version: str


@dataclass
class VMCalculation:
    vmc: VectorialModelConfig
    vmr: VectorialModelResult
    execution_time_s: float
    vectorial_model_backend: str
    vectorial_model_version: str

    @classmethod
    def from_encoded(cls, evmc: EncodedVMCalculation):
        return cls(
            vmc=evmc.vmc,
            vmr=unpickle_from_base64(evmc.evmr),
            execution_time_s=evmc.execution_time_s,
            vectorial_model_backend=evmc.vectorial_model_backend,
            vectorial_model_version=evmc.vectorial_model_version,
        )


def get_saved_model_from_cache(
    vmc: VectorialModelConfig,
) -> Optional[EncodedVMCalculation]:
    hashed_vmc = vmc_to_sha256_digest(vmc=vmc)
    print(f"Looking up hashed vmc {hashed_vmc} in db ...")
    session = get_vm_cache_db_session()
    if session is not None:
        res = session.get(VMCached, hashed_vmc)
        if res:
            evmc = EncodedVMCalculation(
                vmc=vmc,
                evmr=decompress_vmr_string(res.vmr_b64_enc_zip),
                execution_time_s=res.execution_time_s,
                vectorial_model_backend=res.vectorial_model_backend,
                vectorial_model_version=res.vectorial_model_version,
            )
            session.close()
            return evmc
        else:
            print("Not found in db.")
            session.close()
    else:
        print(
            f"get_saved_model_from_cache: No db session, skipping. [hash {hashed_vmc}]"
        )
        return None


def save_model_to_cache(evmc: EncodedVMCalculation) -> None:
    hashed_vmc = vmc_to_sha256_digest(vmc=evmc.vmc)
    print(f"Saving model with hashed vmc {hashed_vmc} in db ...")
    session = get_vm_cache_db_session()
    if session is not None:
        with session.begin():
            model_exists = (
                session.query(VMCached).filter_by(vmc_hash=hashed_vmc).first()
            )
            if not model_exists:
                to_db = VMCached(
                    vmc_hash=hashed_vmc,
                    vmr_b64_enc_zip=compress_vmr_string(evmc.evmr),
                    execution_time_s=evmc.execution_time_s,
                    vectorial_model_backend=evmc.vectorial_model_backend,
                    vectorial_model_version=evmc.vectorial_model_version,
                )
                session.add(to_db)
                session.commit()

    else:
        print(f"save_model_to_cache: No db session, skipping. [hash {hashed_vmc}]")


def rvm_single(
    vmc: VectorialModelConfig,
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ] = PythonModelExtraConfig(print_progress=False),
    vm_cache_dir: Optional[pathlib.Path] = None,
) -> EncodedVMCalculation:
    # attempt to load model if we find it in the database
    if vm_cache_dir:
        # start up a db engine in this process
        initialize_vectorial_model_cache(vectorial_model_cache_dir=vm_cache_dir)
        evmc = get_saved_model_from_cache(vmc=vmc)
        if evmc:
            return evmc

    model_start_time = time.time()
    model_function = None

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
    if vm_cache_dir:
        save_model_to_cache(evmc=evmc)

    return evmc


def rvm_parallel(
    vmc_set: List[VectorialModelConfig],
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ] = PythonModelExtraConfig(print_progress=False),
    parallelism: int = 1,
    vm_cache_dir: Optional[pathlib.Path] = None,
) -> List[VMCalculation]:
    if vm_cache_dir is not None:
        # if we are caching the models, run the unique model configs and cache them,
        # then build the results list like normal
        rvm_parallel_force_unique_models(
            vmc_set=vmc_set,
            extra_config=extra_config,
            parallelism=parallelism,
            vm_cache_dir=vm_cache_dir,
        )
    print("Caching done, running...")

    # The fortran version uses fixed file names for input and output, so running multiple in parallel
    # would clobber each other's input and output files
    if isinstance(extra_config, FortranModelExtraConfig):
        print("Forcing no parallelism for fortran version!")
        parallelism = 1

    pool_start_time = time.time()

    run_vmodel_timed_mappable_func = partial(
        rvm_single, extra_config=extra_config, vm_cache_dir=vm_cache_dir
    )

    with Pool(parallelism) as vm_pool:
        vmcalc_list = vm_pool.map(run_vmodel_timed_mappable_func, vmc_set)

    pool_end_time = time.time()
    print(f"Total results assembly time: {pool_end_time - pool_start_time} seconds")

    return [VMCalculation.from_encoded(x) for x in vmcalc_list]


def rvm_parallel_force_unique_models(
    vmc_set: List[VectorialModelConfig],
    vm_cache_dir: pathlib.Path,
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ] = PythonModelExtraConfig(print_progress=False),
    parallelism: int = 1,
) -> None:
    # The fortran version uses fixed file names for input and output, so running multiple in parallel
    # would clobber each other's input and output files
    if isinstance(extra_config, FortranModelExtraConfig):
        print("Forcing no parallelism for fortran version!")
        parallelism = 1

    vmc_hashes = [vmc_to_sha256_digest(x) for x in vmc_set]
    unique_vmc_hashes = set(vmc_hashes)
    reduced_vmc_set = [
        vmc_set[vmc_hashes.index(unique_hash)] for unique_hash in unique_vmc_hashes
    ]
    print(f"Number of unique models to be run and cached: {len(reduced_vmc_set)}")

    pool_start_time = time.time()

    run_vmodel_timed_mappable_func = partial(
        rvm_single, extra_config=extra_config, vm_cache_dir=vm_cache_dir
    )

    with Pool(parallelism) as vm_pool:
        # run the results and let them be saved in db
        _ = vm_pool.map(run_vmodel_timed_mappable_func, reduced_vmc_set)

    pool_end_time = time.time()
    print(f"Total run time on unique models: {pool_end_time - pool_start_time} seconds")


# # deprecate
# def run_vectorial_models_pooled(
#     vmc_set: List[VectorialModelConfig],
#     extra_config: Union[
#         PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
#     ] = PythonModelExtraConfig(print_progress=False),
#     parallelism: int = 1,
# ) -> List[VMCalculation]:
#     # The fortran version uses fixed file names for input and output, so running multiple in parallel
#     # would clobber each other's input and output files
#     if isinstance(extra_config, FortranModelExtraConfig):
#         print("Forcing no parallelism for fortran version!")
#         parallelism = 1
#
#     pool_start_time = time.time()
#
#     run_vmodel_timed_mappable_func = partial(
#         run_vectorial_model_timed, extra_config=extra_config
#     )
#     with Pool(parallelism) as vm_pool:
#         model_results = vm_pool.map(run_vmodel_timed_mappable_func, vmc_set)
#     pool_end_time = time.time()
#     print(f"Total run time: {pool_end_time - pool_start_time} seconds")
#
#     vmrs = [x[0] for x in model_results]
#     execution_times = [x[1] for x in model_results]
#
#     # TODO: move this to the three backend files: get_python_vectorial_model_version() -> str   etc.
#     if isinstance(extra_config, PythonModelExtraConfig):
#         vectorial_model_backend = "python"
#         vectorial_model_version = impm.version("sbpy")
#     elif isinstance(extra_config, FortranModelExtraConfig):
#         vectorial_model_backend = "fortran"
#         vectorial_model_version = "1.0.0"
#     elif isinstance(extra_config, RustModelExtraConfig):
#         vectorial_model_backend = "rust"
#         vectorial_model_version = "0.1.0"
#
#     vm_calculation_list = [
#         VMCalculation(
#             vmc=vmc,
#             vmr=unpickle_from_base64(vmr),
#             execution_time_s=t.to_value(u.s),  # type: ignore
#             vectorial_model_backend=vectorial_model_backend,
#             vectorial_model_version=vectorial_model_version,
#         )
#         for (vmc, vmr, t) in zip(vmc_set, vmrs, execution_times)
#     ]
#
#     return vm_calculation_list


def store_vmcalculation_list(
    vmcalc_list: List[VMCalculation], out_file: pathlib.Path
) -> None:
    vmcalc_list_pickle = dill.dumps(vmcalc_list)
    with open(out_file, "wb") as f:
        f.write(vmcalc_list_pickle)


def load_vmcalculation_list(in_file: pathlib.Path) -> List[VMCalculation]:
    with open(in_file, "rb") as f:
        pstring = f.read()

    return dill.loads(pstring)


def vmcalc_list_to_dataframe(vmcalc_list: List[VMCalculation]):
    df_list = []

    for vmcalc in vmcalc_list:
        # flatten the nested model_dump dictionary into column names like production.base_q_per_s with json_normalize()
        vmc_df = pd.json_normalize(vmcalc.vmc.model_dump())
        vmr_etc_df = pd.DataFrame(
            [
                {
                    "vmr_base64": pickle_to_base64(vmcalc.vmr),
                    "execution_time_s": vmcalc.execution_time_s,
                    "vectorial_model_backend": vmcalc.vectorial_model_backend,
                    "vectorial_model_version": vmcalc.vectorial_model_version,
                    "vmc_sha256_digest": vmc_to_sha256_digest(vmcalc.vmc),
                }
            ]
        )
        this_df = pd.concat([vmc_df, vmr_etc_df], axis=1)
        df_list.append(this_df)

    return pd.concat(df_list)


def dataframe_to_vmcalc_list(
    df: pd.DataFrame, column_name_separator: str = "."
) -> List[VMCalculation]:
    json_dict_list: List[dict] = []

    # builds a list of dictionaries that each describe a row of the given dataframe.
    # takes column names like production.base_q_per_s and puts them in dict keys json_dict_list['production']['base_q_per_s']
    # by splitting on the column name separator
    for _, row in df.iterrows():
        row_dict = {}
        for column_name, value in row.items():
            assert isinstance(column_name, str)
            keys = column_name.split(column_name_separator)

            cur_dict = row_dict
            for j, key in enumerate(keys):
                if j == len(keys) - 1:
                    cur_dict[key] = value
                else:
                    if key not in cur_dict.keys():
                        cur_dict[key] = {}
                    cur_dict = cur_dict[key]

        json_dict_list.append(row_dict)

    # if we have no time variation, the json value is a NaN float value instead of None, so fix that here
    for jd in json_dict_list:
        if pd.isna(jd["production"]["time_variation"]):
            jd["production"]["time_variation"] = None

    ta = TypeAdapter(VectorialModelConfig)
    vmcl = [
        VMCalculation(
            vmc=ta.validate_python(jd),
            vmr=unpickle_from_base64(jd["vmr_base64"]),
            execution_time_s=jd["execution_time_s"],
            vectorial_model_backend=jd["vectorial_model_backend"],
            vectorial_model_version=jd["vectorial_model_version"],
        )
        for jd in json_dict_list
    ]

    return vmcl

from dataclasses import dataclass
import pathlib
import time
import importlib.metadata as impm
from typing import List, Union
from functools import partial
from multiprocessing import Pool
from numpy import nan

import pandas as pd
import astropy.units as u
import dill
from pydantic import TypeAdapter

from pyvectorial.backends.fortran_version import FortranModelExtraConfig
from pyvectorial.backends.python_version import PythonModelExtraConfig
from pyvectorial.backends.rust_version import RustModelExtraConfig
from pyvectorial.encoding_and_hashing import (
    pickle_to_base64,
    unpickle_from_base64,
    vmc_to_sha256_digest,
)
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


# TODO: use vmc hash to name rust file input and output
def run_vectorial_models_pooled(
    vmc_set: List[VectorialModelConfig],
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ] = PythonModelExtraConfig(print_progress=False),
    parallelism: int = 1,
) -> List[VMCalculation]:
    # The fortran version uses fixed file names for input and output, so running multiple in parallel
    # would clobber each other's input and output files
    if isinstance(extra_config, FortranModelExtraConfig):
        print("Forcing no parallelism for fortran version!")
        parallelism = 1

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

    # TODO: move this to the three backend files: get_python_vectorial_model_version() -> str   etc.
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

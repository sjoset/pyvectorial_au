from dataclasses import dataclass
from typing import TypeAlias

# from functools import partial
# from multiprocessing import Pool

# import pandas as pd
# import dill
# from pydantic import TypeAdapter

from pyvectorial_au.encoding.encoding_and_hashing import unpickle_from_base64
from pyvectorial_au.model_input.vectorial_model_config import VectorialModelConfig
from pyvectorial_au.model_output.vectorial_model_result import VectorialModelResult

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


# def store_vmcalculation_list(
#     vmcalc_list: List[VMCalculation], out_file: pathlib.Path
# ) -> None:
#     vmcalc_list_pickle = dill.dumps(vmcalc_list)
#     with open(out_file, "wb") as f:
#         f.write(vmcalc_list_pickle)
#
#
# def load_vmcalculation_list(in_file: pathlib.Path) -> List[VMCalculation]:
#     with open(in_file, "rb") as f:
#         pstring = f.read()
#
#     return dill.loads(pstring)
#
#
# def vmcalc_list_to_dataframe(vmcalc_list: List[VMCalculation]):
#     df_list = []
#
#     for vmcalc in vmcalc_list:
#         # flatten the nested model_dump dictionary into column names like production.base_q_per_s with json_normalize()
#         vmc_df = pd.json_normalize(vmcalc.vmc.model_dump())
#         vmr_etc_df = pd.DataFrame(
#             [
#                 {
#                     "vmr_base64": pickle_to_base64(vmcalc.vmr),
#                     "execution_time_s": vmcalc.execution_time_s,
#                     "vectorial_model_backend": vmcalc.vectorial_model_backend,
#                     "vectorial_model_version": vmcalc.vectorial_model_version,
#                     "vmc_sha256_digest": vmc_to_sha256_digest(vmcalc.vmc),
#                 }
#             ]
#         )
#         this_df = pd.concat([vmc_df, vmr_etc_df], axis=1)
#         df_list.append(this_df)
#
#     return pd.concat(df_list)


# def dataframe_to_vmcalc_list(
#     df: pd.DataFrame, column_name_separator: str = "."
# ) -> List[VMCalculation]:
#     json_dict_list: List[dict] = []
#
#     # builds a list of dictionaries that each describe a row of the given dataframe.
#     # takes column names like production.base_q_per_s and puts them in dict keys json_dict_list['production']['base_q_per_s']
#     # by splitting on the column name separator
#     for _, row in df.iterrows():
#         row_dict = {}
#         for column_name, value in row.items():
#             assert isinstance(column_name, str)
#             keys = column_name.split(column_name_separator)
#
#             cur_dict = row_dict
#             for j, key in enumerate(keys):
#                 if j == len(keys) - 1:
#                     cur_dict[key] = value
#                 else:
#                     if key not in cur_dict.keys():
#                         cur_dict[key] = {}
#                     cur_dict = cur_dict[key]
#
#         json_dict_list.append(row_dict)
#
#     # if we have no time variation, the json value is a NaN float value instead of None, so fix that here
#     for jd in json_dict_list:
#         if pd.isna(jd["production"]["time_variation"]):
#             jd["production"]["time_variation"] = None
#
#     ta = TypeAdapter(VectorialModelConfig)
#     vmcl = [
#         VMCalculation(
#             vmc=ta.validate_python(jd),
#             vmr=unpickle_from_base64(jd["vmr_base64"]),
#             execution_time_s=jd["execution_time_s"],
#             vectorial_model_backend=jd["vectorial_model_backend"],
#             vectorial_model_version=jd["vectorial_model_version"],
#         )
#         for jd in json_dict_list
#     ]
#
#     return vmcl

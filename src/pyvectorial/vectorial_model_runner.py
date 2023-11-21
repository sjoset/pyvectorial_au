from .vectorial_model_config import VectorialModelConfig
from .vectorial_model_result import VectorialModelResult
from .python_version import PythonModelExtraConfig, run_python_vectorial_model
from .fortran_version import FortranModelExtraConfig, run_fortran_vectorial_model
from .rust_version import RustModelExtraConfig, run_rust_vectorial_model
from typing import Union

"""
    Takes a VectorialModelConfig and runs the python, rust, or fortran models as specified and returns a VectorialModelResult
"""


def run_vectorial_model(
    vmc: VectorialModelConfig,
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ],
) -> VectorialModelResult:
    model_function = None
    if isinstance(extra_config, PythonModelExtraConfig):
        model_function = run_python_vectorial_model
    elif isinstance(extra_config, FortranModelExtraConfig):
        model_function = run_fortran_vectorial_model
    elif isinstance(extra_config, RustModelExtraConfig):
        model_function = run_rust_vectorial_model

    return model_function(vmc, extra_config)  # type: ignore

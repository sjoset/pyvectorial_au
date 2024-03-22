import time
from typing import Tuple, Union, TypeAlias

import astropy.units as u
from astropy.units.quantity import Quantity

from pyvectorial.model_input.vectorial_model_config import VectorialModelConfig
from pyvectorial.model_output.vectorial_model_result import VectorialModelResult
from pyvectorial.backends.python_version import (
    PythonModelExtraConfig,
    run_python_vectorial_model,
)
from pyvectorial.backends.fortran_version import (
    FortranModelExtraConfig,
    run_fortran_vectorial_model,
)
from pyvectorial.backends.rust_version import (
    RustModelExtraConfig,
    run_rust_vectorial_model,
)
from pyvectorial.encoding.encoding_and_hashing import pickle_to_base64

EncodedVectorialModelResult: TypeAlias = str


def run_vectorial_model(
    vmc: VectorialModelConfig,
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ],
) -> VectorialModelResult:
    """
    Takes a VectorialModelConfig and runs the python, rust, or fortran models as specified and returns a VectorialModelResult
    """
    model_function = None
    if isinstance(extra_config, PythonModelExtraConfig):
        model_function = run_python_vectorial_model
    elif isinstance(extra_config, FortranModelExtraConfig):
        model_function = run_fortran_vectorial_model
    elif isinstance(extra_config, RustModelExtraConfig):
        model_function = run_rust_vectorial_model

    return model_function(vmc, extra_config)  # type: ignore


def run_vectorial_model_timed(
    vmc: VectorialModelConfig,
    extra_config: Union[
        PythonModelExtraConfig, FortranModelExtraConfig, RustModelExtraConfig
    ],
) -> Tuple[EncodedVectorialModelResult, Quantity]:
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

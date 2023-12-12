from importlib.metadata import version

__version__ = version("pyvectorial")

from pyvectorial.backends.fortran_version import (
    FortranModelExtraConfig,
    run_fortran_vectorial_model,
    vmr_from_fortran_output,
    fragment_theory_count_from_fortran_output,
    fragment_grid_count_from_fortran_output,
    write_fortran_input_file,
)
from pyvectorial.backends.python_version import (
    PythonModelExtraConfig,
    run_python_vectorial_model,
    vmr_from_sbpy_coma,
)
from pyvectorial.backends.rust_version import (
    RustModelExtraConfig,
    run_rust_vectorial_model,
    vmc_from_rust_output,
    vmr_from_rust_output,
    write_rust_input_file,
)

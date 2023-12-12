from importlib.metadata import version

__version__ = version("pyvectorial")

from pyvectorial.aperture import UncenteredRectangularAperture, total_number_in_aperture
from pyvectorial.calculation_table import (
    add_vmc_columns,
    build_calculation_table,
)
from pyvectorial.column_density_abel import column_density_from_abel
from pyvectorial.input_transforms import apply_input_transform, unapply_input_transform
from pyvectorial.interpolation import (
    interpolate_volume_density,
    interpolate_column_density,
)
from pyvectorial.pickle_encoding import pickle_to_base64, unpickle_from_base64
from pyvectorial.timedependentproduction import (
    TimeDependentProductionType,
    TimeDependentProduction,
)
from pyvectorial.vectorial_model_config import (
    Production,
    Parent,
    Fragment,
    Grid,
    VectorialModelConfig,
    hash_vmc,
)
from pyvectorial.vectorial_model_config_reader import vectorial_model_config_from_yaml
from pyvectorial.vectorial_model_config_writer import vm_config_to_yaml
from pyvectorial.vectorial_model_result import (
    FragmentSputterSpherical,
    FragmentSputterPolar,
    FragmentSputterCartesian,
    VectorialModelResult,
    fragment_sputter_to_cartesian,
    fragment_sputter_to_polar,
    mirror_fragment_sputter,
)
from pyvectorial.vectorial_model_runner import (
    run_vectorial_model,
    run_vectorial_model_timed,
)

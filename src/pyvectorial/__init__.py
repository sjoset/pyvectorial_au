from importlib.metadata import version

__version__ = version("pyvectorial")

from pyvectorial.aperture import UncenteredRectangularAperture, total_number_in_aperture
from pyvectorial.column_density_abel import column_density_from_abel
from pyvectorial.input_transforms import apply_input_transform
from pyvectorial.interpolation import (
    interpolate_volume_density,
    interpolate_column_density,
)
from pyvectorial.encoding_and_hashing import (
    pickle_to_base64,
    unpickle_from_base64,
    vmc_to_sha256_digest,
)
from pyvectorial.timedependentproduction import (
    make_time_dependence_function,
)
from pyvectorial.vectorial_model_config import (
    CometProduction,
    ParentMolecule,
    FragmentMolecule,
    VectorialModelGrid,
    VectorialModelConfig,
)
from pyvectorial.vectorial_model_config_reader import (
    vectorial_model_config_from_yaml,
    vectorial_model_config_from_json,
)
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
from pyvectorial.vectorial_model_calculation import (
    VMCalculation,
    store_vmcalculation_list,
    load_vmcalculation_list,
    vmcalc_list_to_dataframe,
    dataframe_to_vmcalc_list,
)

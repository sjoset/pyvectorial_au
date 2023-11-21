from importlib.metadata import version

__version__ = version("pyvectorial")

from pyvectorial.aperture import UncenteredRectangularAperture, total_number_in_aperture
from pyvectorial.calculation_table import (
    add_vmc_columns,
    build_calculation_table,
)
from pyvectorial.column_density_abel import column_density_from_abel
from pyvectorial.fortran_version import (
    FortranModelExtraConfig,
    run_fortran_vectorial_model,
    vmr_from_fortran_output,
    fragment_theory_count_from_fortran_output,
    fragment_grid_count_from_fortran_output,
    write_fortran_input_file,
)
from pyvectorial.haser_fits import (
    HaserFitResult,
    haser_params_from_full_fit_result,
    haser_q_fit_from_column_density,
    haser_full_fit_from_column_density,
)
from pyvectorial.haser_matplotlib import plot_haser_column_density
from pyvectorial.haser_params import HaserParams, haser_from_vectorial_cd1980
from pyvectorial.input_transforms import apply_input_transform, unapply_input_transform
from pyvectorial.interpolation import (
    interpolate_volume_density,
    interpolate_column_density,
)
from pyvectorial.pickle_encoding import pickle_to_base64, unpickle_from_base64
from pyvectorial.python_version import (
    PythonModelExtraConfig,
    run_python_vectorial_model,
    vmr_from_sbpy_coma,
)
from pyvectorial.rust_version import (
    RustModelExtraConfig,
    run_rust_vectorial_model,
    vmc_from_rust_output,
    vmr_from_rust_output,
    write_rust_input_file,
)
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
from pyvectorial.vm_matplotlib import (
    mpl_mark_collision_sphere,
    mpl_mark_inflection_points,
    mpl_volume_density_plot,
    mpl_volume_density_interpolation_plot,
    mpl_column_density_plot,
    mpl_column_density_interpolation_plot,
    mpl_column_density_plot_3d,
    mpl_fragment_sputter_contour_plot,
    mpl_fragment_sputter_plot,
)
from pyvectorial.vm_plotly import (
    plotly_volume_density_plot,
    plotly_volume_density_interpolation_plot,
    plotly_column_density_plot,
    plotly_column_density_interpolation_plot,
    plotly_column_density_plot_3d,
    plotly_fragment_sputter_contour_plot,
    plotly_fragment_sputter_plot,
    plotly_q_t_plot,
)

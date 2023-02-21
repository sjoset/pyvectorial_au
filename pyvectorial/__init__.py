from .fortran_version import run_fortran_vmodel, get_result_from_fortran

from .timedependentproduction import TimeDependentProduction

from .utils import (
    print_binned_times,
    print_volume_density,
    print_column_density,
    show_fragment_agreement,
    show_aperture_checks,
)

from .vm_matplotlib import (
    mpl_mark_inflection_points,
    mpl_mark_collision_sphere,
    mpl_volume_density_plot,
    mpl_volume_density_interpolation_plot,
    mpl_column_density_plot,
    mpl_column_density_interpolation_plot,
    mpl_column_density_plot_3d,
    mpl_fragment_sputter_contour_plot,
    mpl_fragment_sputter_plot,
)
from .vm_plotly import (
    plotly_volume_density_plot,
    plotly_volume_density_interpolation_plot,
    plotly_column_density_plot,
    plotly_column_density_interpolation_plot,
    plotly_column_density_plot_3d,
    plotly_fragment_sputter_plot,
    plotly_fragment_sputter_contour_plot,
    plotly_q_t_plot,
)

from .vmrunner import run_vmodel

from .vmconfig import VectorialModelConfig, Production, Parent, Fragment, Comet, Grid
from .vmresult import (
    VectorialModelResult,
    get_result_from_coma,
    FragmentSputterPolar,
    FragmentSputterCartesian,
    cartesian_sputter_from_polar,
    mirror_sputter,
)

from .vmconfigread import vm_configs_from_yaml
from .vmconfigwrite import vm_config_to_yaml_file

from .input_transforms import apply_input_transform, unapply_input_transform

from .haser_params import HaserParams, haser_from_vectorial_cd1980
from .haser_fits import (
    HaserScaleLengthSearchResult,
    HaserFitResult,
    haser_q_fit_from_column_density,
    haser_full_fit_from_column_density,
    find_best_haser_scale_lengths_q,
)
from .haser_plots import plot_haser_column_density, haser_search_result_plot

from .hashing import hash_vmc, pickle_to_base64, unpickle_from_base64

from .coma_pickling import coma_from_dill, dill_from_coma

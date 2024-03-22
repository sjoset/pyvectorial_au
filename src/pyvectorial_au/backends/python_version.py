import copy
import logging as log
import sbpy.activity as sba
from sbpy.data import Phys
from dataclasses import dataclass
from pyvectorial_au.pre_model_processing.timedependentproduction import (
    make_time_dependence_function,
)

from pyvectorial_au.model_input.vectorial_model_config import (
    BinnedProductionTimeVariation,
    VectorialModelConfig,
)
from pyvectorial_au.model_output.vectorial_model_result import (
    VectorialModelResult,
    FragmentSputterSpherical,
)


"""
    For interfacing with the python version of the vectorial model
"""


@dataclass
class PythonModelExtraConfig:
    print_progress: bool = False


def parent_molecule_phys_from_vmc(vmc: VectorialModelConfig) -> Phys:
    p = vmc.parent
    p_dict = {
        "tau_d": p.tau_d,
        "tau_T": p.tau_T,
        "v_outflow": p.v_outflow,
        "sigma": p.sigma,
    }

    return Phys.from_dict(p_dict)


def fragment_molecule_phys_from_vmc(vmc: VectorialModelConfig) -> Phys:
    f = vmc.fragment
    f_dict = {"tau_T": f.tau_T, "v_photo": f.v_photo}

    return Phys.from_dict(f_dict)


def run_python_vectorial_model(
    vmc: VectorialModelConfig, extra_config: PythonModelExtraConfig
) -> VectorialModelResult:
    """
    Takes a VectorialModelConfig and builds the time dependence function
    specified, then runs the sbpy vectorial model to return a coma object
    """

    # build parent and fragment inputs
    parent = parent_molecule_phys_from_vmc(vmc=vmc)
    fragment = fragment_molecule_phys_from_vmc(vmc=vmc)

    coma = None
    q_t = None

    # handle binned production as a special case
    if isinstance(vmc.production.time_variation, BinnedProductionTimeVariation):
        # call the binned production constructor that mimics the fortran version
        coma = sba.VectorialModel.binned_production(
            qs=vmc.production.time_variation.q,  # type: ignore
            parent=parent,
            fragment=fragment,
            ts=vmc.production.time_variation.times_at_productions,  # type: ignore
            radial_points=vmc.grid.radial_points,
            angular_points=vmc.grid.angular_points,
            radial_substeps=vmc.grid.radial_substeps,
            print_progress=extra_config.print_progress,
        )
    else:
        if vmc.production.time_variation:
            q_t = make_time_dependence_function(vmc=vmc)
        else:
            q_t = None
            log.info(
                "No valid time dependence specified, assuming steady production of %s",
                vmc.production.base_q,
            )

        coma = sba.VectorialModel(
            base_q=vmc.production.base_q,
            q_t=q_t,
            parent=parent,
            fragment=fragment,
            radial_points=vmc.grid.radial_points,
            angular_points=vmc.grid.angular_points,
            radial_substeps=vmc.grid.radial_substeps,
            print_progress=extra_config.print_progress,
        )

    return VectorialModelResult(
        volume_density_grid=coma.vmr.volume_density_grid,
        volume_density=coma.vmr.volume_density,
        fragment_sputter=FragmentSputterSpherical(
            rs=coma.vmr.fragment_sputter.rs,
            thetas=coma.vmr.fragment_sputter.thetas,
            fragment_density=coma.vmr.fragment_sputter.fragment_density,
        ),
        collision_sphere_radius=coma.vmr.collision_sphere_radius,
        max_grid_radius=coma.vmr.max_grid_radius,
        coma_radius=coma.vmr.coma_radius,
        column_density_grid=coma.vmr.column_density_grid,
        column_density=coma.vmr.column_density,
        volume_density_interpolation=coma.vmr.volume_density_interpolation,
        column_density_interpolation=coma.vmr.column_density_interpolation,
        coma=coma,
    )


def vmr_from_sbpy_coma(coma_orig: sba.VectorialModel) -> VectorialModelResult:
    """Takes a coma object from the sbpy python version and extracts a VectorialModelResult"""
    coma = copy.deepcopy(coma_orig)

    vdg = coma.vmr.volume_density_grid
    vd = coma.vmr.volume_density
    cdg = coma.vmr.column_density_grid
    cd = coma.vmr.column_density
    vdi = coma.vmr.volume_density_interpolation
    cdi = coma.vmr.column_density_interpolation

    fs = FragmentSputterSpherical(
        rs=coma.vmr.fragment_sputter.rs,
        thetas=coma.vmr.fragment_sputter.thetas,
        fragment_density=coma.vmr.fragment_sputter.fragment_density,
    )

    csr = coma.vmr.collision_sphere_radius
    mgr = coma.vmr.max_grid_radius
    cr = coma.vmr.coma_radius

    return VectorialModelResult(
        volume_density_grid=vdg,
        volume_density=vd,
        fragment_sputter=fs,
        collision_sphere_radius=csr,
        max_grid_radius=mgr,
        coma_radius=cr,
        column_density_grid=cdg,
        column_density=cd,
        volume_density_interpolation=vdi,
        column_density_interpolation=cdi,
    )

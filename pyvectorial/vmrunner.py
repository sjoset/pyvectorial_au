
import logging as log
import sbpy.activity as sba
from sbpy.data import Phys
from dataclasses import asdict

from .timedependentproduction import TimeDependentProduction
from .vmconfig import VectorialModelConfig


def run_vmodel(vmc: VectorialModelConfig) -> sba.VectorialModel:

    """
        Takes a VectorialModelConfig and builds the time dependence function
        specified, then runs the sbpy vectorial model to return a coma object
    """
    log.info("Calculating fragment density using vectorial model ...")

    # build parent and fragment inputs
    parent = Phys.from_dict(asdict(vmc.parent))
    fragment = Phys.from_dict(asdict(vmc.fragment))

    coma = None
    q_t = None

    # set up q_t here if we have variable production
    if vmc.production.time_variation_type:
        t_var_type = vmc.production.time_variation_type

        # handle each type of supported time dependence
        if t_var_type == "binned":
            log.debug("Found binned production ...")
            # call the older-style binned production constructor
            coma = sba.VectorialModel.binned_production(qs=vmc.production.params['q_t'],
                                                        parent=parent, fragment=fragment,
                                                        ts=vmc.production.params['times_at_productions'],
                                                        radial_points=vmc.grid.radial_points,
                                                        angular_points=vmc.grid.angular_points,
                                                        radial_substeps=vmc.grid.radial_substeps,
                                                        print_progress=vmc.etc['print_progress']
                                                        )
        elif t_var_type in ["gaussian", "sine wave", "square pulse"]:
            prod_var = TimeDependentProduction(t_var_type)
            q_t = prod_var.create_production(**vmc.production.params)

    # if the binned constructor above hasn't been called, we have work to do
    if coma is None:
        # did we come up with a valid time dependence?
        if q_t is None:
            log.info("No valid time dependence specified, assuming steady production of %s", vmc.production.base_q)

        coma = sba.VectorialModel(base_q=vmc.production.base_q,
                                  q_t=q_t,
                                  parent=parent, fragment=fragment,
                                  radial_points=vmc.grid.radial_points,
                                  angular_points=vmc.grid.angular_points,
                                  radial_substeps=vmc.grid.radial_substeps,
                                  print_progress=vmc.etc['print_progress'])

    return coma

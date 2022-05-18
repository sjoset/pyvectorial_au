
import pickle
import logging as log

from datetime import datetime

from .vmconfigwrite import vm_config_to_yaml_file
from .vmconfig import VectorialModelConfig
from .vmresult import VectorialModelResult


# TODO: deprecated
# def save_vmodel(vmc: VectorialModelConfig, vmodel: dict, base_outfile_name: str) -> None:
#
#     """
#         Saves parameters of vmodel run along with the results in a separate pickled file
#     """
#     config_save = base_outfile_name + ".yaml"
#     pickle_file = base_outfile_name + ".vm"
#
#     # TODO: find a way to get version cleanly
#     # vmc.etc['version'] = __version__
#     vmc.etc['pyv_date_of_run'] = datetime.now().strftime("%m %d %Y")
#     vmc.etc['pyv_coma_pickle'] = pickle_file
#
#     log.info("Writing parameters to %s ...", config_save)
#     vm_config_to_yaml_file(vmc, config_save)
#
#     log.info("Writing model results to %s ...", pickle_file)
#     with open(pickle_file, 'wb') as picklejar:
#         pickle.dump(vmodel, picklejar)


def save_results(vmc: VectorialModelConfig, vmr: VectorialModelResult, base_outfile_name) -> None:

    config_save = base_outfile_name + ".yaml"
    pickle_file = base_outfile_name + ".vmr"

    # TODO: find a way to get version cleanly
    # vmc.etc['version'] = __version__
    vmc.etc['pyv_date_of_run'] = datetime.now().strftime("%m %d %Y")
    vmc.etc['pyv_coma_pickle'] = pickle_file

    log.info("Writing parameters to %s ...", config_save)
    vm_config_to_yaml_file(vmc, config_save)

    log.info("Writing model results to %s ...", pickle_file)
    with open(pickle_file, 'wb') as picklejar:
        pickle.dump(vmr, picklejar)


import pickle
import logging as log

from datetime import datetime

from .parameters import dump_parameters_to_file
from .vm_config import vm_config_to_yaml


def save_vmodel(vmc, vmodel, base_outfile_name):

    """
        Saves parameters of vmodel run along with the results in a separate pickled file
    """
    config_save = base_outfile_name + ".yaml"
    pickle_file = base_outfile_name + ".vm"

    # TODO: find a way to do this cleanly
    # vmc.etc['version'] = __version__
    vmc.etc['pyv_date_of_run'] = datetime.now().strftime("%m %d %Y")
    vmc.etc['pyv_coma_pickle'] = pickle_file

    log.info("Writing parameters to %s ...", config_save)
    vm_config_to_yaml(vmc, config_save)

    log.info("Writing model results to %s ...", pickle_file)
    with open(pickle_file, 'wb') as picklejar:
        pickle.dump(vmodel, picklejar)


def save_vmodel_old(input_yaml, vmodel, base_outfile_name):

    """
        Saves parameters of vmodel run along with the results in a separate pickled file
    """
    parameters_file = base_outfile_name + ".yaml"
    pickle_file = base_outfile_name + ".vm"

    input_yaml['pyvectorial_info'] = {}
    # TODO: find a way to do this cleanly
    # input_yaml['pyvectorial_info']['version'] = __version__
    input_yaml['pyvectorial_info']['date_run'] = datetime.now().strftime("%m %d %Y")
    input_yaml['pyvectorial_info']['vmodel_pickle'] = pickle_file

    log.info("Writing parameters to %s ...", parameters_file)
    dump_parameters_to_file(parameters_file, input_yaml)

    log.info("Writing model results to %s ...", pickle_file)
    with open(pickle_file, 'wb') as picklejar:
        pickle.dump(vmodel, picklejar)


def pickle_vmodel(vmodel, base_outfile_name):

    """
        Just dumps vmodel of finished run to the filename given appended with .vm
    """

    pickle_file = base_outfile_name + ".vm"
    log.info("Writing model results to %s ...", pickle_file)
    with open(pickle_file, 'wb') as picklejar:
        pickle.dump(vmodel, picklejar)

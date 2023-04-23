import copy
import yaml
import pathlib
import astropy.units as u
from dataclasses import asdict

from .vectorial_model_config import VectorialModelConfig


"""
    Functionality for taking a VectorialModelConfig and producing a YAML file storing the config.
    Paired with vm_configs_from_yaml, this can save/load model configurations for inspection or re-use later.
"""


def vm_config_to_yaml(vmc: VectorialModelConfig, filepath: pathlib.Path) -> None:
    dict_to_write = _strip_of_units(vmc)

    with open(filepath, "w") as stream:
        try:
            yaml.safe_dump(dict_to_write, stream)
        except yaml.YAMLError as exc:
            print(exc)


"""
    The python-supplied YAML library doesn't currently work with astropy's unit system, so we have to
    convert the quantities manually for storage before writing.
"""


def _strip_of_units(vmc_orig: VectorialModelConfig) -> dict:
    # TODO: remove the float() calls once the bug in pyyaml writing decimal numbers is fixed
    # https://github.com/yaml/pyyaml/issues/255

    vmcd = asdict(copy.deepcopy(vmc_orig))

    vmcd["production"]["base_q"] = float(vmcd["production"]["base_q"].to(1 / u.s).value)

    p = vmcd["production"]
    # handle the time-variation types
    if p.get("time_variation_type") is not None:
        t_var_type = p["time_variation_type"]
        if t_var_type == "binned":
            p["params"]["q_t"] = list(
                map(float, list(p["params"]["q_t"].to(1 / u.s).value))
            )
            p["params"]["times_at_productions"] = list(
                map(float, list(p["params"]["times_at_productions"].to(u.day).value))
            )
        elif t_var_type == "gaussian":
            p["params"]["amplitude"] = float(p["params"]["amplitude"].to(1 / u.s).value)
            p["params"]["t_max"] = float(p["params"]["t_max"].to(u.hour).value)
            p["params"]["std_dev"] = float(p["params"]["std_dev"].to(u.hour).value)
        elif t_var_type == "sine wave":
            p["params"]["amplitude"] = float(p["params"]["amplitude"].to(1 / u.s).value)
            p["params"]["period"] = float(p["params"]["period"].to(u.hour).value)
            p["params"]["delta"] = float(p["params"]["delta"].to(u.hour).value)
        elif t_var_type == "square pulse":
            p["params"]["amplitude"] = float(p["params"]["amplitude"].to(1 / u.s).value)
            p["params"]["t_start"] = float(p["params"]["t_start"].to(u.hour).value)
            p["params"]["duration"] = float(p["params"]["duration"].to(u.hour).value)

    vmcd["parent"]["v_outflow"] = float(
        vmcd["parent"]["v_outflow"].to(u.km / u.s).value
    )
    vmcd["parent"]["tau_T"] = float(vmcd["parent"]["tau_T"].to(u.s).value)
    vmcd["parent"]["tau_d"] = float(vmcd["parent"]["tau_d"].to(u.s).value)
    vmcd["parent"]["sigma"] = float(vmcd["parent"]["sigma"].to(u.cm**2).value)

    # fragment
    vmcd["fragment"]["v_photo"] = float(
        vmcd["fragment"]["v_photo"].to(u.km / u.s).value
    )
    vmcd["fragment"]["tau_T"] = float(vmcd["fragment"]["tau_T"].to(u.s).value)

    return vmcd

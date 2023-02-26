import copy
import yaml

import logging as log
import astropy.units as u
import numpy as np

from itertools import product

from .vmconfig import VectorialModelConfig, Production, Parent, Fragment, Comet, Grid
from .input_transforms import apply_input_transform

from typing import List

"""
    Functionality for taking a YAML file and constructing a list of VectorialModelConfigs specified
    by the contents of the YAML
"""


def vm_configs_from_yaml(filepath: str) -> List[VectorialModelConfig]:
    """
    Takes a filename with yaml configuration in it and returns
    a list of all configs if any of the keys 'allowed_variations'
    is defined as a list
    """

    vmc = _vm_config_from_yaml(filepath, init_ratio=False)

    # holds a list of all combinations of changing parameters in vmc,
    # values given as lists in the yaml instead of single values
    varying_parameters = []

    # list of VectorialModelConfigs built from these changing parameters
    psets = []

    # Look at these inputs and build all possible combinations for running
    allowed_variations = [vmc.production.base_q, vmc.parent.tau_d, vmc.fragment.tau_T]

    q_t_variation = None
    # check for outburst variations that may or may not exist
    # Gaussian, sine wave, square pulse
    for variable_key in ["t_max", "delta", "t_start"]:
        if variable_key in vmc.production.params:
            allowed_variations.append(vmc.production.params[variable_key])
            q_t_variation = variable_key

    # check if it is a list, and if not, make it a list of length 1
    # build list of lists of changing values for input to itertools.product
    for av in allowed_variations:
        # already a list, add it
        if isinstance(av.value, np.ndarray):
            varying_parameters.append(av)
        else:
            # Single value specified so make a 1-element list
            varying_parameters.append([av])

    for element in product(*varying_parameters):
        # Make copy to append to our list
        new_vmc = copy.deepcopy(vmc)

        # Update this copy with the values we are varying
        # format is a tuple with format (base_q, parent_tau_d, fragment_tau_T)
        new_vmc.production.base_q = copy.copy(element[0])
        new_vmc.parent.tau_d = copy.copy(element[1])
        new_vmc.parent.tau_T = copy.copy(element[1]) * new_vmc.parent.T_to_d_ratio
        new_vmc.fragment.tau_T = copy.copy(element[2])

        if q_t_variation:
            new_vmc.production.params[q_t_variation] = copy.deepcopy(element[3])

        # we can apply these transforms now that vmc.parent.tau_T is filled in
        apply_input_transform(new_vmc)

        psets.append(new_vmc)

    return psets


def _vm_config_from_yaml(
    filepath: str, init_ratio: bool = True
) -> VectorialModelConfig:
    input_yaml = _read_yaml_from_file(filepath)

    production = _production_from_yaml(input_yaml)
    del input_yaml["production"]

    parent = _parent_from_yaml(input_yaml, init_ratio)
    del input_yaml["parent"]

    fragment = _fragment_from_yaml(input_yaml)
    del input_yaml["fragment"]

    comet = _comet_from_yaml(input_yaml)
    del input_yaml["comet"]

    grid = _grid_from_yaml(input_yaml)
    del input_yaml["grid"]

    vmc = VectorialModelConfig(
        production=production,
        parent=parent,
        fragment=fragment,
        comet=comet,
        grid=grid,
        etc=input_yaml.get("etc"),
    )

    # # defaults
    if vmc.etc is None:
        vmc.etc = dict()
    vmc.etc.setdefault("print_progress", False)

    return vmc


# TODO: break this up into get_gaussian_production(input_yaml: dict) -> Production, etc. etc.
def _production_from_yaml(input_yaml: dict) -> Production:
    # Read the production config from input_yaml and apply units

    # Pop things off of p and pass the rest onto the params
    p = input_yaml["production"]
    base_q = p.pop("base_q", 0.0) * (1 / u.s)

    # TODO: rewrite without all the repitition
    t_var_type = p.pop("time_variation_type", None)
    if t_var_type == "binned":
        log.debug("Found binned production ...")
        bin_required = set(["q_t", "times_at_productions"])
        has_reqs = bin_required.issubset(p["params"].keys())
        if not has_reqs:
            print(
                "Required keys for binned production not found in production params section!"
            )
            print(f"Need {list(bin_required)}.")
            print("Exiting.")
            exit(1)
        p["params"]["q_t"] *= 1 / u.s
        p["params"]["times_at_productions"] *= u.day
    elif t_var_type == "gaussian":
        log.debug("Found gaussian production ...")
        gauss_reqs = set(["amplitude", "t_max", "std_dev"])
        has_reqs = gauss_reqs.issubset(p["params"].keys())
        if not has_reqs:
            print(
                "Required keys for gaussian production not found in production params section!"
            )
            print(f"Need {list(gauss_reqs)}.")
            print("Exiting.")
            exit(1)
        p["params"]["amplitude"] *= 1 / u.s
        p["params"]["t_max"] *= u.hour
        p["params"]["std_dev"] *= u.hour
        log.debug(
            "Amplitude: %s, t_max: %s, std_dev: %s",
            p["params"]["amplitude"],
            p["params"]["t_max"],
            p["params"]["std_dev"],
        )
    elif t_var_type == "sine wave":
        log.debug("Found sine wave production ...")
        sine_reqs = set(["amplitude", "period", "delta"])
        has_reqs = sine_reqs.issubset(p["params"].keys())
        if not has_reqs:
            print(
                "Required keys for sine wave production not found in production params section!"
            )
            print(f"Need {list(sine_reqs)}.")
            print("Exiting.")
            exit(1)
        p["params"]["amplitude"] *= 1 / u.s
        p["params"]["period"] *= u.hour
        p["params"]["delta"] *= u.hour
        log.debug(
            "Amplitude: %s, period: %s, delta: %s",
            p["params"]["amplitude"],
            p["params"]["period"],
            p["params"]["delta"],
        )
    elif t_var_type == "square pulse":
        log.debug("Found square pulse production ...")
        sq_reqs = set(["amplitude", "t_start", "duration"])
        has_reqs = sq_reqs.issubset(p["params"].keys())
        if not has_reqs:
            print(
                "Required keys for square pulse production not found in production params section!"
            )
            print(f"Need {list(sq_reqs)}.")
            print("Exiting.")
            exit(1)
        p["params"]["amplitude"] *= 1 / u.s
        p["params"]["t_start"] *= u.hour
        p["params"]["duration"] *= u.hour
        log.debug(
            "Amplitude: %s, t_start: %s, duration: %s",
            p["params"]["amplitude"],
            p["params"]["t_start"],
            p["params"]["duration"],
        )
    params = p.get("params", dict())

    return Production(base_q=base_q, time_variation_type=t_var_type, params=params)


def _parent_from_yaml(input_yaml: dict, init_ratio: bool) -> Parent:
    p = input_yaml["parent"]

    # if we specify either of p[tau_d] or p[T_to_d_ratio] as a list in the yaml, this errors out
    #  so init_ratio = false avoids the multiplication.  The user has to fill in tau_T later if this
    #  function is called directly
    tau_T = 0 * u.s
    if init_ratio:
        tau_T = (p["tau_d"] * p["T_to_d_ratio"] * u.s,)

    return Parent(
        name=p.get("name"),
        v_outflow=p["v_outflow"] * u.km / u.s,
        tau_d=p["tau_d"] * u.s,
        tau_T=tau_T,
        sigma=p["sigma"] * u.cm**2,
        T_to_d_ratio=p["T_to_d_ratio"],
    )


def _fragment_from_yaml(input_yaml: dict) -> Fragment:
    f = input_yaml["fragment"]

    return Fragment(
        name=f.get("name"), v_photo=f["v_photo"] * u.km / u.s, tau_T=f["tau_T"] * u.s
    )


def _comet_from_yaml(input_yaml: dict) -> Comet:
    p = input_yaml["comet"]

    return Comet(
        name=p.get("name"),
        rh=p.get("rh", 1.0) * u.AU,
        delta=p.get("delta", 1.0) * u.AU,
        transform_method=p.get("transform_method"),
        transform_applied=p.get("transform_applied", False),
    )


def _grid_from_yaml(input_yaml: dict) -> Grid:
    g = input_yaml["grid"]

    return Grid(
        radial_points=g.get("radial_points"),
        angular_points=g.get("angular_points"),
        radial_substeps=g.get("radial_substeps"),
    )


def _read_yaml_from_file(filepath: str) -> dict:
    """Read YAML file from disk and return dictionary"""
    with open(filepath, "r") as stream:
        try:
            param_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return param_yaml

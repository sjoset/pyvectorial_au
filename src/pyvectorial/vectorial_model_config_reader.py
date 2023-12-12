import yaml
import pathlib

import logging as log
import astropy.units as u
from typing import Optional

from pyvectorial.vectorial_model_config import (
    VectorialModelConfig,
    Production,
    Parent,
    Fragment,
    Grid,
)


def vectorial_model_config_from_yaml(
    filepath: pathlib.Path,
) -> Optional[VectorialModelConfig]:
    input_yaml = _read_yaml_from_file(filepath)

    if input_yaml is None:
        return None

    production = _production_from_yaml(input_yaml)
    del input_yaml["production"]

    parent = _parent_from_yaml(input_yaml)
    del input_yaml["parent"]

    fragment = _fragment_from_yaml(input_yaml)
    del input_yaml["fragment"]

    grid = _grid_from_yaml(input_yaml)
    del input_yaml["grid"]

    vmc = VectorialModelConfig(
        production=production,
        parent=parent,
        fragment=fragment,
        grid=grid,
    )

    return vmc


# TODO: break this up into get_gaussian_production(input_yaml: dict) -> Production, etc. etc.
def _production_from_yaml(input_yaml: dict) -> Production:
    # Read the production config from input_yaml and apply units

    # Pop things off of p and pass the rest onto the params
    p = input_yaml["production"]
    base_q = p.pop("base_q", 0.0) * (1 / u.s)

    # TODO: rewrite without all the repetition
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


def _parent_from_yaml(input_yaml: dict) -> Parent:
    p = input_yaml["parent"]

    return Parent(
        v_outflow=p["v_outflow"] * u.km / u.s,
        tau_d=p["tau_d"] * u.s,
        tau_T=p["tau_T"] * u.s,
        sigma=p["sigma"] * u.cm**2,  # type: ignore
    )


def _fragment_from_yaml(input_yaml: dict) -> Fragment:
    f = input_yaml["fragment"]

    return Fragment(v_photo=f["v_photo"] * u.km / u.s, tau_T=f["tau_T"] * u.s)


def _grid_from_yaml(input_yaml: dict) -> Grid:
    g = input_yaml["grid"]

    return Grid(
        radial_points=g["radial_points"],
        angular_points=g["angular_points"],
        radial_substeps=g["radial_substeps"],
        parent_destruction_level=g["parent_destruction_level"],
        fragment_destruction_level=g["fragment_destruction_level"],
    )


def _read_yaml_from_file(filepath: pathlib.Path) -> Optional[dict]:
    """Read YAML file from disk and return dictionary"""
    with open(filepath, "r") as stream:
        try:
            param_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            param_yaml = None
            log.info("Reading file %s resulted in yaml error: %s", filepath, exc)

    return param_yaml

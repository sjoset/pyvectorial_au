
import copy
import yaml

import logging as log
import astropy.units as u
import numpy as np

from .vmconfig import VectorialModelConfig, Production, Parent, Fragment, Comet, Grid


def vm_config_from_yaml(filepath: str, init_ratio: bool = True) -> VectorialModelConfig:

    input_yaml = _read_yaml_from_file(filepath)

    production = _production_from_yaml(input_yaml)
    del input_yaml['production']

    parent = _parent_from_yaml(input_yaml, init_ratio)
    del input_yaml['parent']

    fragment = _fragment_from_yaml(input_yaml)
    del input_yaml['fragment']

    comet = _comet_from_yaml(input_yaml)
    del input_yaml['comet']

    grid = _grid_from_yaml(input_yaml)
    del input_yaml['grid']

    vmc = VectorialModelConfig(
            production=production,
            parent=parent,
            fragment=fragment,
            comet=comet,
            grid=grid,
            etc=input_yaml['etc']
            )

    # defaults
    vmc.etc['print_progress'] = input_yaml['etc'].get('print_progress', False)

    vmc_orig = copy.deepcopy(vmc)
    _apply_transform_method(vmc)

    return vmc, vmc_orig


def _production_from_yaml(input_yaml: dict) -> Production:

    # Read the production config from input_yaml and apply units

    # Pop things off of p and pass the rest onto the params
    p = input_yaml['production']
    base_q = p.pop('base_q', 0.0) * (1/u.s)

    # TODO: rewrite without all the repitition
    t_var_type = p.pop('time_variation_type', None)
    if t_var_type == "binned":
        log.debug("Found binned production ...")
        bin_required = set(['q_t', 'times_at_productions'])
        has_reqs = bin_required.issubset(p['params'].keys())
        if not has_reqs:
            print("Required keys for binned production not found in production params section!")
            print(f"Need {list(bin_required)}.")
            print("Exiting.")
            exit(1)
        p['params']['q_t'] *= (1/u.s)
        p['params']['times_at_productions'] *= u.day
    elif t_var_type == "gaussian":
        log.debug("Found gaussian production ...")
        gauss_reqs = set(['amplitude', 't_max', 'std_dev'])
        has_reqs = gauss_reqs.issubset(p['params'].keys())
        if not has_reqs:
            print("Required keys for gaussian production not found in production params section!")
            print(f"Need {list(gauss_reqs)}.")
            print("Exiting.")
            exit(1)
        p['params']['amplitude'] *= (1/u.s)
        p['params']['t_max'] *= u.hour
        p['params']['std_dev'] *= u.hour
        log.debug("Amplitude: %s, t_max: %s, std_dev: %s", p['params']['amplitude'], p['params']['t_max'], p['params']['std_dev'])
    elif t_var_type == "sine wave":
        log.debug("Found sine wave production ...")
        sine_reqs = set(['amplitude', 'period', 'delta'])
        has_reqs = sine_reqs.issubset(p['params'].keys())
        if not has_reqs:
            print("Required keys for sine wave production not found in production params section!")
            print(f"Need {list(sine_reqs)}.")
            print("Exiting.")
            exit(1)
        p['params']['amplitude'] *= (1/u.s)
        p['params']['period'] *= u.hour
        p['params']['delta'] *= u.hour
        log.debug("Amplitude: %s, period: %s, delta: %s", p['params']['amplitude'], p['params']['period'], p['params']['delta'])
    elif t_var_type == "square pulse":
        log.debug("Found square pulse production ...")
        sq_reqs = set(['amplitude', 't_start', 'duration'])
        has_reqs = sq_reqs.issubset(p['params'].keys())
        if not has_reqs:
            print("Required keys for square pulse production not found in production params section!")
            print(f"Need {list(sq_reqs)}.")
            print("Exiting.")
            exit(1)
        p['params']['amplitude'] *= (1/u.s)
        p['params']['t_start'] *= u.hour
        p['params']['duration'] *= u.hour
        log.debug("Amplitude: %s, t_start: %s, duration: %s", p['params']['amplitude'], p['params']['t_start'], p['params']['duration'])
    params = p.get('params', None)

    return Production(base_q=base_q, time_variation_type=t_var_type, params=params)


def _parent_from_yaml(input_yaml: dict, init_ratio: bool) -> Parent:
    p = input_yaml['parent']

    # if we specify either of p[tau_d] or p[T_to_d_ratio] as a list in the yaml, this errors out
    #  so init_ratio = false avoids the multiplication.  The user has to fill in tau_T later!
    tau_T = 0 * u.s
    if init_ratio:
        tau_T=p['tau_d'] * p['T_to_d_ratio'] * u.s,

    return Parent(
            name=p.get('name'),
            v_outflow=p['v_outflow'] * u.km/u.s,
            tau_d=p['tau_d'] * u.s,
            tau_T=tau_T,
            sigma=p['sigma'] * u.cm**2,
            T_to_d_ratio=p['T_to_d_ratio']
            )


def _fragment_from_yaml(input_yaml: dict) -> Fragment:
    f = input_yaml['fragment']

    return Fragment(
            name=f.get('name'),
            v_photo=f['v_photo'] * u.km/u.s,
            tau_T=f['tau_T'] * u.s
            )


def _comet_from_yaml(input_yaml: dict) -> Comet:
    p = input_yaml['comet']

    return Comet(
            name=p.get('name'),
            rh=p.get('rh', 1.0) * u.AU,
            delta=p.get('delta', 1.0) * u.AU,
            transform_method=p.get('transform_method'),
            transform_applied=p.get('transform_applied', False)
            )


def _grid_from_yaml(input_yaml: dict) -> Grid:
    g = input_yaml['grid']

    return Grid(
            radial_points=g.get('radial_points'),
            angular_points=g.get('angular_points'),
            radial_substeps=g.get('radial_substeps')
            )


def _apply_transform_method(vmc: VectorialModelConfig) -> None:

    log.info("Current transform state: %s", vmc.comet.transform_applied)

    if vmc.comet.transform_applied:
        log.info("Attempted to apply transform more than once, skipping")
        return

    # if none specified, nothing to do
    if vmc.comet.transform_method is None:
        log.info("No valid tranformation of input data specified, no transform applied")
        return

    log.info("Transforming input parameters using method %s", vmc.comet.transform_method)

    if vmc.comet.transform_method == 'cochran_schleicher_93':
        # TODO: shouldn't this affect vmc.parent.tau_T instead of the ratio?
        # TODO: reread the paper to make sure
        log.info("Reminder: cochran_schleicher_93 overwrites v_outflow of parent")
        rh = vmc.comet.rh.to(u.AU).value
        sqrh = np.sqrt(rh)

        v_old = copy.deepcopy(vmc.parent.v_outflow)
        tau_d_old = copy.deepcopy(vmc.parent.tau_d)
        ttod_old = copy.deepcopy(vmc.parent.T_to_d_ratio)

        vmc.parent.v_outflow = (0.85/sqrh) * u.km/u.s
        vmc.parent.tau_d *= rh**2
        vmc.parent.T_to_d_ratio = vmc.parent.tau_T / vmc.parent.tau_d

        log.info("Effect of transform at %s AU:", rh)
        log.info("Parent outflow: %s --> %s", v_old, vmc.parent.v_outflow)
        log.info("Parent tau_d: %s --> %s", tau_d_old, vmc.parent.tau_d)
        log.info("Total to dissociative ratio: %s --> %s", ttod_old, vmc.parent.T_to_d_ratio)

    elif vmc.comet.transform_method == 'festou_fortran':
        rh = vmc.comet.rh.to(u.AU).value
        ptau_d_old = copy.deepcopy(vmc.parent.tau_d)
        ptau_T_old = copy.deepcopy(vmc.parent.tau_T)
        ftau_T_old = copy.deepcopy(vmc.fragment.tau_T)
        vmc.parent.tau_d *= rh**2
        vmc.parent.tau_T *= rh**2
        vmc.fragment.tau_T *= rh**2
        log.info("\tParent tau_d: %s --> %s", ptau_d_old, vmc.parent.tau_d)
        log.info("\tParent tau_T: %s --> %s", ptau_T_old, vmc.parent.tau_T)
        log.info("\tFragment tau_T: %s --> %s", ftau_T_old, vmc.fragment.tau_T)

    else:
        log.info("Invalid transform method specified, skipping")
        return

    vmc.comet.transform_applied = True


def _read_yaml_from_file(filepath: str) -> dict:
    """ Read YAML file from disk and return dictionary """
    with open(filepath, 'r') as stream:
        try:
            param_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return param_yaml

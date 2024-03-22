import os
import pytest
import numpy as np
import astropy.units as u
from pyvectorial.pre_model_processing.timedependentproduction import (
    make_time_dependence_function,
)
from pyvectorial.model_input.vectorial_model_config_reader import (
    vectorial_model_config_from_yaml,
)


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


# binned production fixtures
@pytest.fixture
def single_run_binned_yaml(rootdir):
    return os.path.join(rootdir, "input/single_run_binned.yaml")


@pytest.fixture
def vmc_binned(single_run_binned_yaml):
    return vectorial_model_config_from_yaml(single_run_binned_yaml)


# gaussian fixtures
@pytest.fixture
def single_run_gaussian_yaml(rootdir):
    return os.path.join(rootdir, "input/single_run_gaussian.yaml")


@pytest.fixture
def vmc_gaussian(single_run_gaussian_yaml):
    return vectorial_model_config_from_yaml(single_run_gaussian_yaml)


@pytest.fixture
def q_t_gaussian(vmc_gaussian):
    return make_time_dependence_function(vmc_gaussian)


# sine wave fixtures
@pytest.fixture
def single_run_sine_yaml(rootdir):
    return os.path.join(rootdir, "input/single_run_sine.yaml")


@pytest.fixture
def vmc_sine(single_run_sine_yaml):
    return vectorial_model_config_from_yaml(single_run_sine_yaml)


@pytest.fixture
def q_t_sine(vmc_sine):
    return make_time_dependence_function(vmc_sine)


# square pulse fixtures
@pytest.fixture
def single_run_square_pulse_yaml(rootdir):
    return os.path.join(rootdir, "input/single_run_square_pulse.yaml")


@pytest.fixture
def vmc_square_pulse(single_run_square_pulse_yaml):
    return vectorial_model_config_from_yaml(single_run_square_pulse_yaml)


@pytest.fixture
def q_t_square_pulse(vmc_square_pulse):
    return make_time_dependence_function(vmc_square_pulse)


# binned tests
def test_binned_q_first(vmc_binned):
    assert np.isclose(vmc_binned.production.time_variation.q[0], 7.0e25 / u.s)


def test_binned_q_last(vmc_binned):
    assert np.isclose(vmc_binned.production.time_variation.q[-1], 1.0e28 / u.s)


def test_binned_times_first(vmc_binned):
    assert np.isclose(
        vmc_binned.production.time_variation.times_at_productions[0], 35 * u.day  # type: ignore
    )


def test_binned_times_last(vmc_binned):
    assert np.isclose(
        vmc_binned.production.time_variation.times_at_productions[-1], 5 * u.day  # type: ignore
    )


# gaussian tests
def test_gaussian_amplitude(vmc_gaussian):
    assert np.isclose(vmc_gaussian.production.time_variation.amplitude, 3.0e28 / u.s)


def test_gaussian_std_dev(vmc_gaussian):
    assert np.isclose(vmc_gaussian.production.time_variation.std_dev, 20.0 * u.hour)  # type: ignore


def test_gaussian_t_max(vmc_gaussian):
    assert np.isclose(vmc_gaussian.production.time_variation.t_max, 24.0 * u.hour)  # type: ignore


def test_gaussian_q_at_t_max(q_t_gaussian):
    assert np.isclose(q_t_gaussian((24.0 * u.hour).to_value(u.s)), 3.0e28)  # type: ignore


# time for half max is mu +/- sqrt(2 ln(2)) * sigma
def test_gaussian_q_at_half_max_one(q_t_gaussian):
    assert np.isclose(
        q_t_gaussian(((24.0 - np.sqrt(2 * np.log(2)) * 20.0) * u.hour).to_value(u.s)),
        1.5e28,
    )


def test_gaussian_q_at_half_max_two(q_t_gaussian):
    assert np.isclose(
        q_t_gaussian(((24.0 + np.sqrt(2 * np.log(2)) * 20.0) * u.hour).to_value(u.s)),
        1.5e28,
    )


# sine tests
def test_sine_amplitude(vmc_sine):
    assert np.isclose(vmc_sine.production.time_variation.amplitude, 3.0e28 / u.s)


def test_sine_period(vmc_sine):
    assert np.isclose(vmc_sine.production.time_variation.period, 20.0 * u.hour)  # type: ignore


def test_sine_delta(vmc_sine):
    assert np.isclose(vmc_sine.production.time_variation.delta, 3.0 * u.hour)  # type: ignore


def test_sine_q_t_at_delta(q_t_sine):
    assert np.isclose(q_t_sine((-3.0 * u.hour).to_value(u.s)), 0.0)  # type: ignore


def test_sine_q_t_at_first_max_in_past(q_t_sine):
    assert np.isclose(q_t_sine((2.0 * u.hour).to_value(u.s)), 3.0e28)  # type: ignore


def test_sine_q_t_at_first_min_in_past(q_t_sine):
    assert np.isclose(q_t_sine((12.0 * u.hour).to_value(u.s)), -3.0e28)  # type: ignore


# square pulse tests
def test_square_pulse_amplitude(vmc_square_pulse):
    assert np.isclose(
        vmc_square_pulse.production.time_variation.amplitude, 4.0e28 / u.s
    )


def test_square_pulse_duration(vmc_square_pulse):
    assert np.isclose(vmc_square_pulse.production.time_variation.duration, 5.0 * u.hour)  # type: ignore


def test_square_pulse_t_start(vmc_square_pulse):
    assert np.isclose(vmc_square_pulse.production.time_variation.t_start, 10.0 * u.hour)  # type: ignore


def test_q_t_square_pulse_during(q_t_square_pulse):
    assert np.isclose(q_t_square_pulse((7.0 * u.hour).to_value(u.s)), 4.0e28)  # type: ignore


def test_q_t_square_pulse_before(q_t_square_pulse):
    assert np.isclose(q_t_square_pulse((15.0 * u.hour).to_value(u.s)), 0.0)  # type: ignore


def test_q_t_square_pulse_after(q_t_square_pulse):
    assert np.isclose(q_t_square_pulse((0.0 * u.hour).to_value(u.s)), 0.0)  # type: ignore

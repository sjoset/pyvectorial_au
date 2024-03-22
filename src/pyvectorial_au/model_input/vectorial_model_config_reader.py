import yaml
import json
import pathlib
import logging as log
from typing import Optional

from pyvectorial.model_input.vectorial_model_config import VectorialModelConfig


def vectorial_model_config_from_yaml(
    filepath: pathlib.Path,
) -> Optional[VectorialModelConfig]:
    with open(filepath, "r") as stream:
        try:
            input_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            log.info("Reading file %s resulted in yaml error: %s", filepath, exc)
            return None

    return VectorialModelConfig(**input_yaml)


def vectorial_model_config_from_json(
    filepath: pathlib.Path,
) -> Optional[VectorialModelConfig]:
    with open(filepath, "r") as f:
        try:
            input_json = json.load(f)
        except json.JSONDecodeError as msg:
            log.info("Reading file %s resulted in json error: %s", filepath, msg)
            return None

    return VectorialModelConfig(**input_json)

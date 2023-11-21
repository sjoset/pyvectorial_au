import hashlib
import pprint

from .vectorial_model_config import VectorialModelConfig

"""
    If we have a large number of models stored, we may want to quickly search whether or not a given
    VectorialModelConfig has been run already.  If we generate a unique hash based on the
    VectorialModelConfig and save that along with the model results, we can compare hashes to see if
    this particular VectorialModelConfig has been run already.
"""


# Our function for generating a deterministic hash based on the vmc
def hash_vmc(vmc: VectorialModelConfig):
    return hashlib.sha256(pprint.pformat(vmc).encode("utf-8")).hexdigest()

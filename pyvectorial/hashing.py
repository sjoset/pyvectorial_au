
import hashlib
import pprint
import codecs
import dill as pickle

from .vmconfig import VectorialModelConfig

"""
    If we have a large number of models stored, we may want to quickly search whether or not a given
    VectorialModelConfig has been run already.  If we generate a unique hash based on the
    VectorialModelConfig and save that along with the model results, we can compare hashes to see if
    this particular VectorialModelConfig has been run already.
"""

# Our function for generating a deterministic hash based on the vmc
def hash_vmc(vmc: VectorialModelConfig):

    return hashlib.sha256(pprint.pformat(vmc).encode('utf-8')).hexdigest()


# TODO: move these two to pickle_encoding.py
# Serialize an object to a base64 string for storage
def pickle_to_base64(obj) -> str:

    return codecs.encode(pickle.dumps(obj), 'base64').decode()


# Restore object from base64 string encoding
def unpickle_from_base64(p_str: str):

    return pickle.loads(codecs.decode(p_str.encode(), 'base64'))

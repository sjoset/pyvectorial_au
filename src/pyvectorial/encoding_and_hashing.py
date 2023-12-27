import hashlib
import dill as pickle
import codecs

from pyvectorial.vectorial_model_config import VectorialModelConfig


# Serialize an object to a base64 string for storage
def pickle_to_base64(obj) -> str:
    return codecs.encode(pickle.dumps(obj), "base64").decode()


# Restore object from base64 string encoding
def unpickle_from_base64(p_str: str):
    return pickle.loads(codecs.decode(p_str.encode(), "base64"))


# return a string of the sha256 hex digest of the given vmc
def vmc_to_sha256_digest(vmc: VectorialModelConfig) -> str:
    h = hashlib.sha256()
    h.update(vmc.model_dump_json().encode())

    return h.hexdigest()

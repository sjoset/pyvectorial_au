import zlib
import hashlib
import dill as pickle
import codecs
from typing import Any, TypeAlias

from pyvectorial_au.model_input.vectorial_model_config import VectorialModelConfig

VectorialModelConfigHash: TypeAlias = str


# Serialize an object to a base64 string for storage
def pickle_to_base64(obj: Any) -> str:
    return codecs.encode(pickle.dumps(obj), "base64").decode()


# Restore object from base64 string encoding
def unpickle_from_base64(p_str: str) -> Any:
    return pickle.loads(codecs.decode(p_str.encode(), "base64"))


# return a string of the sha256 hex digest of the given vmc
def vmc_to_sha256_digest(vmc: VectorialModelConfig) -> VectorialModelConfigHash:
    h = hashlib.sha256()
    h.update(vmc.model_dump_json().encode())

    return h.hexdigest()


def compress_vmr_string(vmr: str) -> bytes:
    return zlib.compress(vmr.encode())


def decompress_vmr_string(vmr_compressed: bytes) -> str:
    return zlib.decompress(vmr_compressed).decode()

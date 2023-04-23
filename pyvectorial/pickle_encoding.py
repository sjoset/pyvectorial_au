import dill as pickle
import codecs


# Serialize an object to a base64 string for storage
def pickle_to_base64(obj) -> str:
    return codecs.encode(pickle.dumps(obj), "base64").decode()


# Restore object from base64 string encoding
def unpickle_from_base64(p_str: str):
    return pickle.loads(codecs.decode(p_str.encode(), "base64"))

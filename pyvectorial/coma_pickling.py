
import dill as pickle


"""
    Using dill, we can serialize our completed models that contain arbitrary time dependence functions,
    which the stock python pickler fails on
"""


def dill_from_coma(coma, coma_file) -> None:
    with open(coma_file, 'wb') as comapicklefile:
        pickle.dump(coma, comapicklefile)


def coma_from_dill(coma_file: str):
    with open(coma_file, 'rb') as coma_dill:
        return pickle.load(coma_dill)

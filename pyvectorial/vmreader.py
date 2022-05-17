
import pickle
from .vmresult import VectorialModelResult


def read_vmodel(vmodelfile):

    with open(vmodelfile, 'rb') as vmp:
        return pickle.load(vmp)


def read_results(vmresult_file: str) -> VectorialModelResult:
    with open(vmresult_file, 'rb') as vmrp:
        return pickle.load(vmrp)

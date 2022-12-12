
import pickle
from .vmresult import VectorialModelResult


def read_results(vmresult_file: str) -> VectorialModelResult:
    with open(vmresult_file, 'rb') as vmrp:
        return pickle.load(vmrp)

import numpy as np


def dump_array(array : np.ndarray, format="%.2f") -> str:
    """
    Short string representation of array for debugging
    """
    if array.ndim > 1:
        return "["+",".join(dump_array(x) for x in array)+"]"
    return "["+",".join(format%x for x in array)+"]"

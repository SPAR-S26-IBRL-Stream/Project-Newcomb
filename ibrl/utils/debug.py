import numpy as np
from numpy.typing import NDArray


def dump_array(array : NDArray[np.float64], format="%.2f") -> str:
    """
    Short string representation of array for debugging
    """
    if array.ndim > 1:
        return "["+",".join(dump_array(x) for x in array)+"]"
    return "["+",".join(format%x for x in array)+"]"

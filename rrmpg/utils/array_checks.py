#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>
"""Implementation of various checks that will be performed on inputs."""

import numpy as np
import pandas as pd

from numba import njit

@njit
def check_for_negatives(arr):
    """Check if array contains negative number.
    
    Numba optimized function to check if a numpy array containes a negative
    value. Returns, whenever the first negative function is found.
    
    Args:
        arr: Numpy array
    
    Returns:
        True, if the array contains at least on negative number and False, if
        the array contains no negative number.
    """
    for val in arr:
        if val < 0:
            return True
    return False


def validate_array_input(arr, dtype, arr_name):
    """Check if array has correct type and is numerical.

    This function checks if the input is either a list, numpy.ndarray or
    pandas.Series of numerical values, converts it to a numpy.ndarray and
    throws an error in case of incorrect data.

    Args:
        arr: Array of data
        dtype: One of numpy's dtypes
        arr_name: String specifing the variable name, so that the error
            message can be adapted correctly.

    Returns:
        A as numpy.ndarray converted array of values with a datatype
        specified in the input argument.

    Raises:
        ValueError: In case non-numerical data is passed
        TypeError: If the error is neither a list, a numpy.ndarray nor a 
            pandas.Series

    """
    # Check for correct data type
    if isinstance(arr, (list, np.ndarray, pd.Series)):
        # Try to convert as numpy array
        try:
            arr = np.array(arr, dtype=dtype).flatten()
        except:
            msg = ["The data in the parameter array '{}'".format(arr_name),
                   " must be purely numerical."]
            raise ValueError("".join(msg))
    else:
        msg = ["The array {} must be either a list, ".format(arr_name),
               "numpy.ndarray or pandas.Series"]
        raise TypeError("".join(msg))
    
    # return converted array
    return arr

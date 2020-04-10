import numpy as np


def get_dt(t):
    arg_dt = 20 if len(t) >= 20 else len(t)
    dt = np.median(np.diff(t[:arg_dt]))
    return dt

def shift_array(arr, shift, fill_value=False):
    """
    Shifts array on axis 0 filling the shifted values with fill_value
    Positive shift is to the right, negative to the left
    """

    result = np.empty_like(arr)
    if shift > 0:
        result[:shift, ...] = fill_value
        result[shift:, ...] = arr[:-shift, ...]
    elif shift < 0:
        result[shift:, ...] = fill_value
        result[:shift, ...] = arr[-shift:, ...]
    else:
        result = arr
    return result
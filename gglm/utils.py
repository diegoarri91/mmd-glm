import numpy as np


def get_dt(t):
    arg_dt = 20 if len(t) >= 20 else len(t)
    dt = np.median(np.diff(t[:arg_dt]))
    return dt


def searchsorted(t, s, side='left'):
    '''
    Uses np.searchsorted but handles numerical round error with care
    such that returned index satisfies
    t[i-1] < s <= t[i]
    np.searchsorted(side='right') doesn't properly handle the equality sign
    on the right side
    '''
    s = np.atleast_1d(s)
    arg = np.searchsorted(t, s, side=side)

    if len(t) > 1:
        dt = get_dt(t)
        s_ = (s - t[0]) / dt
        round_s = np.round(s_, 0)
        mask_round = np.isclose(s_, np.round(s_, 0)) & (round_s >= 0) & (round_s < len(t))
        if side == 'left':
            arg[mask_round] = np.array(round_s[mask_round], dtype=int)
        elif side == 'right':
            arg[mask_round] = np.array(round_s[mask_round], dtype=int) + 1
    else:
        s_ = s - t[0]
        mask = np.isclose(s - t[0], 0.)# & (round_s >= 0) & (round_s < len(t))
        arg[mask] = np.array(s_[mask], dtype=int)

    if len(arg) == 1:
        arg = arg[0]

    return arg


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
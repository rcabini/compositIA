import numpy as np

# Windowing of input data
def windower(data,wmin,wmax):
    """
    windower function, gets in input a numpy array,
    the minimum HU value and the maximum HU value.
    It returns a windowed numpy array with the same dimension of data
    containing values between 0 and 255
    """
    dump = data.copy()
    dump[dump>=wmax] = wmax
    dump[dump<=wmin] = wmin
    dump -= wmin
    w = wmax - wmin
    return dump / w * 255


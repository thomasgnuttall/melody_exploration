import numpy as np

def get_stationary_point(a, n_best=1, min_max='min', up_a=None, low_a=None, mask_mx=None):
    """
    Return minimum or maximum in time series <a>

    :param a: time series array
    :type a: numpy.array
    :param n_best: 1 = return best (highest min/max), 2 = return second best (second highest min/max)
    :type n_best: int
    :param min_max: 'min' or 'max'
    :type min_max: str
    :param up_a: Upper bound of values considered, any elements of <a> not below this threshold are not considered, default None
    :type up_a: float or None
    :param low_a: Lower bound of values considered, any elements of <a> not above this threshold are not considered, default None
    :type low_a: float or None
    :param mask_mx: If not None only consider elements in <a> with these indices, default None
    :type mask_mx: np.array or None

    :return: Index of a corresponding to stationary point requested
    :rtype: int
    """
    if not n_best > 0:
        raise ValueError('<n_best> a positive integer')

    if not isinstance(a, np.ndarray):
        a = np.array(a)
    
    if not up_a:
        up_a = max(a)
    if not low_a:
        low_a = min(a)

    valid_idx = np.where((a >= low_a) & (a <= up_a))[0]

    if not (mask_mx is None):
        valid_idx = np.intersect1d(valid_idx, mask_mx)

    if min_max=='min':
        return valid_idx[np.argsort(a[valid_idx])[n_best-1]]
    elif min_max=='max':
        return valid_idx[np.argsort(a[valid_idx])[-n_best]]
    else:
        raise NameError("<min_max> must equal 'min' or 'max'")


def find_nearest(array, value, index=True):
    """
    Find the closest element of <array> to <value>

    :param array: array of values
    :type array: numpy.array
    :param value: value to check
    :type value: float
    :param index: True or False, return index or value in <array> of closest element?
    :type index: bool

    :return: index/value of element in <array> closest to <value>
    :rtype: number
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx if index else array[idx]
import itertools
import logging
import os 
import operator

from kneed import KneeLocator
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import stumpy

from src.chains import kde_cluster_1d


def get_logger(name):
    logging.basicConfig(format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger(__name__)

def get_stationary_point(a, top_n=1, min_max='min', up_a=None, low_a=None, mask_mx=None):
    """
    Return minimum or maximum in time series <a>

    :param a: time series array
    :type a: numpy.array
    :param top_n: 1 = return best (highest min/max), 2 = return top two best etc... (second highest min/max)
    :type top_n: int
    :param min_max: 'min' or 'max'
    :type min_max: str
    :param up_a: Upper bound of values considered, any elements of <a> not below this threshold are not considered, default None
    :type up_a: float or None
    :param low_a: Lower bound of values considered, any elements of <a> not above this threshold are not considered, default None
    :type low_a: float or None
    :param mask_mx: If not None only consider elements in <a> with these indices, default None
    :type mask_mx: np.array or None

    :return: Indices of stationary points requested
    :rtype: np.array if top_n > 1 else int
    """
    if not top_n > 0:
        raise ValueError('<top_n> a positive integer')
    
    if min_max not in ['min', 'max']:
        raise NameError("<min_max> must equal 'min' or 'max'")
    
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
        idx = np.argsort(a[valid_idx])[:top_n]
        if top_n == 1:
            idx = idx[0]
        return valid_idx[idx]
    elif min_max=='max':
        idx = np.array(list(reversed(np.argsort(a[valid_idx])[-top_n:])))
        if top_n == 1:
            idx = idx[0]
        return valid_idx[idx]


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


def convert(s):
    """
    Evaluate string equation, <s>
    
    :param s: Either a string float or fraction
    :type s: str

    :return: float of evaluated string (eg convert('3/4')==0.75)
    :rtype: float
    """
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)


def get_peaks_and_magnitudes(x, y, kwargs={}):
    """
    Get peaks in x and y data using scipy find_peaks

    :param x: x values
    :type x: numpy.array
    :param y: y values
    :type y: numpy.array
    :param kwargs: Dict of keyword arguments for scipy.signal.find_peaks
    :type kwargs: dict

    :return: [(peak location in x, y value at peak),...] for all peaks
    :rtype: list(tuples)
    """
    peak_ix, peak_arr = find_peaks(y, **kwargs)
    x_y = [(X,Y) for X,Y in zip(x[peak_ix], y[peak_ix])]
    return x_y


def find_nearest(array, value, index=False):
    """
    Find nearest element in <array> to <value>

    :param array: numpy array
    :type array: np.array
    :param array: Number
    :type array: float/int
    :param index: return index/value (True/False)
    :param index: bool

    :return: index or value of element in <array> closest to <value>
    :rtype: float/int
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx if index else array[idx]


def myround(x, base=5):
    return base * round(x/base)


def get_keyword_path(d, keywords, all_keywords=True, with_d=True):
    """
    Return list of files/sub-directories in directory, <d> that have <keywords> in the name

    :param d: directory to search in
    :type d: str
    :param keywords: Keywords to search for
    :type keywords: str/list
    :param all_keywords: If <keyword> is a list whether to return all
        files/sub-directories containing all or just one of keywords (True/False)
    :type all_keywords: bool
    :param with_d: If true, return paths including <d>
    :type with_d: bool

    :return: list of paths with keywords
    :rtype: list
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    all_obj = os.listdir(d)
    if all_keywords:
        matches = [x for x in all_obj if all([k in x for k in keywords])]
    else:
        matches = [x for x in all_obj if any([k in x for k in keywords])]
    return [os.path.join(d, m) for m in matches] if with_d else matches


def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    arr = arr.astype('float')
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)[0], arr[detected_minima]


def get_non_silence(pitch, m_secs):
    return np.array([0 if 0 in pitch[i:i+int(m_secs/0.0029)] else 1 for i in range(len(pitch))])


def get_minima(rec, mp, m_secs, sig_percentile, bandwidth, remove_silent=True):
    minima = detect_local_minima(mp[:, 0])

    # filter out patterns that include silence
    non_silence = get_non_silence(rec.pitch, m_secs)
    if remove_silent:
        ns_i = np.array([i for i,x in enumerate(minima[0]) if non_silence[x]])
        if ns_i.size == 0:
            return []
        minima = [minima[0][ns_i], minima[1][ns_i]]

    # threshold
    th_i = [i for i,x in enumerate(minima[1]) if x <= np.percentile(minima[1], sig_percentile)]
    minima = minima[0][th_i]
    
    # cluster nearby
    clustered = [x for x in kde_cluster_1d(minima, bandwidth=bandwidth)]
    minima_labelled = zip(clustered, minima)

    # Take most similar candidate from each cluster
    it = itertools.groupby(minima_labelled, operator.itemgetter(0))
    cluster_ss = []
    for k, si in it:
        cluster_ss.append(min([s[1] for s in si]))

    return cluster_ss


def search_and_cluster(minima, rec, m_secs, N=None):
    """
    For list of subsequence start points, <minima> apply get_subsequence_cluster()
    """
    out_seq = []
    out_sig = []
    for m in minima:
        pats, sigs = get_subsequence_cluster(ss=m, rec=rec, m_secs=m_secs, N=N)
        out_seq.append(pats)
        out_sig.append(sigs)
    return out_seq, out_sig


def get_subsequence_cluster(ss, rec, m_secs, N=None):
    """
    For subsequence <rec>.pitch[<ss>:<ss>+<m_secs>], compute distance between itself and 
    every other subsequence in <rec>.pitch. Return top <N> unique subsequences. If N is not 
    specified, determine cutoff using Knee method (https://raghavan.usc.edu//papers/kneedle-simplex11.pdf)
    """
    stmass = stumpy.core.mass(rec.pitch[ss:ss+int(m_secs/0.0029)], rec.pitch, normalize=False)
    
    # Take 50000 most similar
    mass_index = np.array(sorted(enumerate(stmass), key=lambda y: y[1]))[:50000]

    # Order by index for KDE
    mass_index = np.array(sorted(mass_index, key=lambda y: y[0]))
    
    # Cluster subsequences start points that are part odf the same motif using KDE
    labels = np.array([x for x in kde_cluster_1d(mass_index[:,0], bandwidth=m_secs/(6*0.0029))])
    labelled = np.dstack([mass_index[:,0], mass_index[:,1], labels])[0]
    
    # Take most similar candidate from each cluster
    it = itertools.groupby(labelled, operator.itemgetter(2))
    cluster_ss = []
    for k, si in it:
        grouped_array = [s for s in si]
        gasort = sorted(grouped_array, key=lambda y: y[1])
        cluster_ss.append((gasort[0][0],gasort[0][1]))

    # sort back to most similar
    ranked = sorted(cluster_ss, key=lambda y: y[1])
    x = range(len(ranked))
    y = [x[1] for x in ranked]
    if not N:
        kn = KneeLocator(x, y, curve='concave', direction='increasing')
        N = kn.knee
    
    pats = [x[0] for x in ranked][:N]
    sigs = [x[1] for x in ranked][:N]

    return pats, sigs


def get_timestamp(secs, divider='-'):
    """
    Convert seconds into timestamp

    :param secs: seconds
    :type secs: int
    :param divider: divider between minute and second, default "-"
    :type divider: str

    :return: timestamp
    :rtype: str
    """
    minutes = int(secs/60)
    seconds = round(secs%60, 2)
    return f'{minutes}min{divider}{seconds}sec'


def interpolate_below_length(arr, val, gap):
    """
    Interpolate gaps of value, <val> of 
    length equal to or shorter than <gap> in <arr>
    
    :param arr: Array to interpolate
    :type arr: np.array
    :param val: Value expected in gaps to interpolate
    :type val: number
    :param gap: Maximum gap length to interpolate, gaps of <val> longer than <g> will not be interpolated
    :type gap: number

    :return: interpolated array
    :rtype: np.array
    """
    s = np.copy(arr)
    is_zero = s == val
    cumsum = np.cumsum(is_zero).astype('float')
    diff = np.zeros_like(s)
    diff[~is_zero] = np.diff(cumsum[~is_zero], prepend=0)
    for i,d in enumerate(diff):
        if d <= gap:
            s[int(i-d):i] = np.nan
    interp = pd.Series(s).interpolate(method='linear', axis=0)\
                         .ffill()\
                         .bfill()\
                         .values
    return interp

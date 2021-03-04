from collections.abc import Iterable 
import math
from numbers import Number
import numpy as np 

from src.utils import get_peaks_and_magnitudes, find_nearest

def force_range(n, u, l):
    """
    Force <n> to be in range <l>:<u> by doubling
    or halving
    """
    if n > u:
        return n/2
    elif n < l:
        return n*2
    else:
        return n


def octavize_pitch(pitch, upper_bound, lower_bound):
    """
    Ensure only pitch values we care about for computing 
    pitch distribution exist in array.
        - 0 values are discarded
        - pitches outside of range are halved or doubled
            to fall within range

    :param pitch: array of pitch values
    :type pitch: np.array
    :param upper_bound: Higher bound of pitch range to search
    :type upper_bound: np.arraye
    :param lower_bound: lower bound of pitch range to search
    :type upper_bound: float

    :return: array of pitch values
    :rtype: np.array
    """
    assert upper_bound > lower_bound, \
        "Upper bound must be greater than lower bound if specified"

    vfunc = np.vectorize(lambda y: force_range(y, upper_bound, lower_bound))
    p = vfunc(pitch)
    p1 = p[np.where(p < upper_bound)]
    return p1[np.where(p1 > lower_bound)]


def locate_pitch_peaks(bins, vals, pitch_dict, epsilon=3, kwargs={}):
    """
    Find peaks in input histogram corresponding to those
    expected in <pitch_dict>. If expected peaks in <pitch_dict> are a range,
    return highest peak in that range. If it is a number, return closest peak
    to that number

    :param bins: Bins of pitch histogram
    :type bins: np.array
    :param vals: Values of pitch histogram
    :type vals: np.array
    :param pitch_dict: Dict of (peak_name: expected_freq),
        expected_freq can be a value corresponding to an expected location for this peak_name
        or it can be a range of values [r1, r2] where r1 is less than r2
    :type pitch_dict: pitch_dict
    :param epsilon: Value in units of <bins>. Allowed room for error in values specified in <pitch_dict>
        e.g if a peak in <pitch_dict> is expected at 145 and epsilon was 3, look in the range 142:148 for this peak
    :param kwargs: Dict of kwargs for scipy.signal.find_peaks
    :type kwargs: dict

    :return: Dict of peak names (from pitch_dict.keys()): peak x values (same units as bins)
    """
    # epsilon. Small bit of leeway on frequency bands 
    peaks_and_densities = get_peaks_and_magnitudes(bins, vals, kwargs)

    # order maintained in all_svaras
    # Find frequency centers for each svara
    peaks_dict = {}
    for s, ef in pitch_dict.items():
        if isinstance(ef, Iterable):
            assert len(ef) == 2, \
                f"Range of peak, '{s}' should be specified as (upper bound, lower bound)"
            p1, p2 = ef
            these = [(p,d,i) for i,(p,d) in enumerate(peaks_and_densities) \
                             if p >= p1-epsilon and p <= p2+epsilon]
            
            # Take pitch of peak in range with largest y value
            winner = sorted(these, key=lambda y: -y[1])[0]
            peaks_dict[s] = winner[0]
            
            # remove peak from original array
            del peaks_and_densities[winner[2]]

        elif isinstance(ef, Number):
            peaks = [p for p,d in peaks_and_densities]
            idx = find_nearest(peaks, ef, index=True)
            peaks_dict[s] = peaks[idx]
            del peaks_and_densities[idx]

    return peaks_dict


def pitch_dict_octave_extend(pitch_dict, lower_prefix='_oct', higher_prefix='_OCT'):
    """
    Add to a dict of {pitch name: pitch frequency values} with the lower and higher octave
    :param pitch_dict: {pitch name: pitch value}
    :type pitch_dict: dict(str, float)
    :param lower_prefix: Prefix to append to lower octave name
    :type lower_prefix: str
    :param higher_prefix: Prefix to append to higher octave name
    :type higher_prefix: str
    """
    d = {}
    for k,v in pitch_dict.items():
        d[k] = v
        d[k+lower_prefix] = v/2
        d[k+higher_prefix] = v*2
    return d


def pitch_to_cents(p, tonic):
    """
    Convert pitch value, <p> to cents above <tonic>.

    :param p: Pitch value in Hz
    :type p: float
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Pitch value, <p> in cents above <tonic>
    :rtype: float
    """
    return 1200*math.log(p/tonic, 2) if p else None


def cents_to_pitch(c, tonic):
    """
    Convert cents value, <c> to pitch in Hz

    :param c: Pitch value in cents above <tonic>
    :type c: float/int
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Pitch value, <c> in Hz 
    :rtype: float
    """
    return (2**(c/1200))*tonic


def pitch_seq_to_cents(pseq, tonic):
    """
    Convert sequence of pitch values to sequence of 
    cents above <tonic> values

    :param pseq: Array of pitch values in Hz
    :type pseq: np.array
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Sequence of original pitch value in cents above <tonic>
    :rtype: np.array
    """
    return np.vectorize(lambda y: pitch_to_cents(y, tonic))(pseq)



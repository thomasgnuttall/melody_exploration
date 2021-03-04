import numpy as np
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
import stumpy 

from src.utils import find_nearest

def get_chains(mp_left, mp_right, mask=[], m=None, min_length=2, sort=True, sort_key=lambda y: -len(y)):
    """
    Get iterable of all chains identified by matrix profile. 

    :param mp_left: left indices from matrix profile index
    :type mp_left: numpy.array
    :param mp_right: right indices from matrix profile index
    :type mp_right: numpy.array
    :param mask: Array of 0s and 1s. Chains that include sequences traversing 0 regions are not returned
    :type mask: numpy.array
    :param m: subsequence legnth in matrix profile, only relevant if using <mask>, default None
    :type m: int
    :param min_length: Chains below this legnth are not returned
    :type min_length: int
    :param sort: if True, sort output by <sort_key>
    :type sort: bool
    :param sort_key: key for sort, default to order of descending length of chain
    :type sort_key: lambda function

    :return: array of all chains, each chain is an array of subsequence start points
    :rtype: numpy.array
    """
    m = 0 if not m else m
    all_chain_set, _ = stumpy.allc(mp_left, mp_right)
    av_chain_len = np.mean([len(x) for x in all_chain_set])
    # TODO: speed this up
    all_chain_set = [x for x in all_chain_set if len(x) >= min_length and not any([0 in mask[i:i+m] for i in x])]
    if sort:
        all_chain_set = sorted(all_chain_set, key=sort_key)
    return all_chain_set


def average_chain(a):
    return [np.mean(x) for x in a]


def kde_cluster_1d(a, bandwidth, kernel='gaussian', res=5000):
    """
    1 dimensional clustering of points in <a> using maximum of
    Kernel Density

    :param a: Array of values to be clustered
    :type a: numpy.array
    :param bandwidth: Bandwidth of KernelDensity
    :type bandwidth: float
    :param kernel: kernel in KDE, default 'gaussian'
    :type kernel: str
    :param res: Number of points to interpolate before maximum identification, default 5000
    :type res: int
    
    :return: Array of cluster labels for each element in <a>
    :rtype: numpy.array
    """
    a_i = range(len(a))
    a = np.array(a).reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)\
            .fit(a)
    s = np.linspace(0, max(a), num=res)
    e = kde.score_samples(s.reshape(-1, 1))

    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    
    maximums = [s[m] for m in ma]

    # Cluster each element to its nearest maximum
    return [find_nearest(maximums, v) for v in a]


def get_longest(all_chains_clustered):
    """
    Extract a representative chain from each cluster in <all_chains_clustered>

    :param all_chains_clustered: Array of chains and cluster label to be clustered, (chain, cluster label)
    :type all_chains_clustered: numpy.array([np.array, int])
    
    :return: dict of {cluster label: candidate chain}
    :rtype: dict
    """
    # Extract a sequence for all chains
    cluster_candidates = {}
    clusters = set(all_chains_clustered[:,1])
    for c in clusters:
        this_cluster = all_chains_clustered[np.where(all_chains_clustered[:,1]==c)]
        this_clust_sort = sorted(this_cluster, key=lambda y: -len(y))
        cluster_candidates[c] = this_clust_sort[0][0]
    return cluster_candidates


def get_clustered_chains(chains, feature_func, cluster_func, candidate_func, feature_func_kwargs={}, cluster_func_kwargs={}, candidate_func_kwargs={}):
    """
    Cluster a list of sequence chains using specified functions

    :param chains: array of chains, where chain is array of subsequence start points
    :type chains: numpy.array
    :param feature_func: Fucntion that takes as its first argument chains and returns an array of equal lenght of features for <cluster_func>
    :type feature_func: python function
    :param cluster_func: Takes output array of feature_func and clusters, returning array of cluster allocations
    :type cluster_func: python function
    :param candidate_func: Takes array of (chain, cluster allocation) and returns a dict of {cluster:chain} where chain is selected by <candidate_func>
    :type candidate_func: python function
    :param cluster_kwargs: dict of arguments for <cluster>
    :type cluster_kwargs: dict
    :param feature_func_kwargs: dict of arguments for <feature_func>
    :type feature_func_kwargs: dict
    :param candidate_func_kwargs: dict of arguments for <candidate_func>
    :type candidate_func_kwargs: dict
    

    :return: {cluster:chain}
    :rtype: dict
    """
    # AVERAGE BEST MATCHES
    matches_av = feature_func(chains, **feature_func_kwargs)

    # Cluster chains
    cluster_indices = cluster_func(matches_av, **cluster_func_kwargs)
    all_chains_clustered = np.array(list(zip(chains, cluster_indices)), dtype=object)

    # pick candidate
    cluster_candidates = candidate_func(all_chains_clustered, **candidate_func_kwargs)

    return cluster_candidates
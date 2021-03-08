import itertools
import operator
import os
import shutil

import numpy as np
from scipy.sparse import csr_matrix
import stumpy
from sklearn.cluster import DBSCAN
import tqdm

from src.io import (
    load_json, load_yaml, load_tonic, load_pitch_contour, 
    create_if_not_exists, write_matrix_profile, load_matrix_profile)
from src.pitch.transformation import (
    octavize_pitch, locate_pitch_peaks, pitch_dict_octave_extend, 
    pitch_seq_to_cents, pitch_to_cents, cents_to_pitch)
from src.utils import get_logger, get_keyword_path, interpolate_below_length

logger = get_logger(__name__)

def rel_path(p):
    """Convert path relative to module into absolute path"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), p))

SVARA_LOOKUP = rel_path('../../conf/svara_lookup.yaml')
#SVARA_LOOKUP='conf/svara_lookusp.yaml'

SVARA_CENT = rel_path('../../conf/svara_cents.yaml')
#SVARA_CENT='conf/svara_cents.yaml'

class Recording:
    """Recording Class"""
    def __init__(self, rec_dir, gap_interp=0.06, svara_lookup_path=SVARA_LOOKUP, svara_cent_path=SVARA_CENT):

        # Args
        self.rec_dir = rec_dir
        self.svara_lookup_path = svara_lookup_path
        self.svara_cent_path = svara_cent_path

        # Metadata
        self.metadata_path = get_keyword_path(self.rec_dir, '.json')[0]
        self.metadata = load_json(self.metadata_path)
        self.__unpack_json(self.metadata)

        # Paths
        pitch_vocal_path = get_keyword_path(self.rec_dir, 'pitch-vocal.txt')
        pitch_vocal_path = pitch_vocal_path[0] if pitch_vocal_path else None

        pitch_mix_path = get_keyword_path(self.rec_dir, 'pitch.txt')
        pitch_mix_path = pitch_mix_path[0] if pitch_mix_path else None

        audio_path = get_keyword_path(self.rec_dir, '.mp3')
        audio_path = audio_path[0] if audio_path else None

        tonic_path = get_keyword_path(self.rec_dir, 'tonic')
        tonic_path = tonic_path[0] if tonic_path else None


        self.pitch_vocal_path = pitch_vocal_path if pitch_vocal_path else None
        self.pitch_mix_path = pitch_mix_path if pitch_mix_path else None
        self.audio_path = audio_path if audio_path else None
        self.tonic_path = tonic_path if tonic_path else None
        
        # Get tonic
        self.tonic = load_tonic(self.tonic_path)

        # Get freq ratios
        self.arohana, self.avorahana = self.__get_svaras(self.raaga, self.svara_lookup_path)
        self.svaras = set(self.arohana + self.avorahana)

        # Get cent mappings
        self.svara_cents = self.__get_svara_cents(self.svaras, self.svara_cent_path)
        self.svara_freq = {k:cents_to_pitch(v, self.tonic) for k,v in self.svara_cents.items()}

        # Get time, pitch
        if self.pitch_vocal_path:
            self.times, self.pitch = load_pitch_contour(self.pitch_vocal_path)
        elif self.pitch_mix_path:
            logger.warning('No isolated vocal pitch track using mix instead')
            self.times, self.pitch = load_pitch_contour(self.pitch_mix_path)
        else:
            logger.warning('No pitch tracks found')
        
        # Interpolate gaps of length equal to or less than 60ms
        self.pitch = interpolate_below_length(self.pitch, 0, int(gap_interp*0.001/0.0029))
        self.silence_mask = self.pitch == 0

        # In cents
        self.pitch_cents = pitch_seq_to_cents(self.pitch, self.tonic)
        

    def self_matrix_profile(self, m, seconds=True, res=0.0029, cache=False, cache_dir=None):
        """
        Compute the self matrix profile of self.pitch
        with subsequence of length=<m>.

        :param m: subsequence length
        :type m: int
        :param seconds: If True, m is specified in seconds and 
            converted to number of elements of length <res> seconds
        :type seconds: bool
        :param res: Only relevant if <seconds> - length of each timestep in sequence
        :type res: float
        :param cache: If True, store a local cache of this matrix profile in <cache_dir>
        :type cache: bool
        :param cache_dir: Only relevant if <cache>, mp will be cahced locally in this location, defaults to <self.rec_dir>/.matrix_profile/
        :type cache_dir: bool
        
        :return: self matrix profile
        :rtype: numpy.array
        """
        if not cache_dir:
            cache_dir = os.path.join(f'{self.rec_dir}', '.matrix_profile/','')

        if seconds==True:
            # Pattern length in seconds
            m = int(m/res)

        mp_name = f'__mp_self_m{str(m)}_seconds{str(seconds)}'
        this_mp = self.__getattr__(mp_name)

        if not this_mp is None:
            if cache:
                path = os.path.join(cache_dir, f'{mp_name}.tsv')
                create_if_not_exists(path)
                write_matrix_profile(this_mp, path)

            return self.__getattr__(mp_name)
        
        mp_ex = self.__mp_exists(cache_dir, f'{mp_name}.tsv')

        if not mp_ex is None:
            setattr(self, mp_name, mp_ex)
            return mp_ex

        mp = stumpy.stump(self.pitch, m=m, normalize=False)

        setattr(self, mp_name, mp)

        if cache:
            path = os.path.join(cache_dir, f'{mp_name}.tsv')
            create_if_not_exists(path)
            write_matrix_profile(mp, path)

        return self.__getattr__(mp_name)

    def get_self_similarity(m, seconds=True, res=0.0029, cache=False, cache_dir=None):
        """
        Compute the self similarity profile of self.pitch
        with subsequence of length=<m>.

        :param m: subsequence length
        :type m: int
        :param seconds: If True, m is specified in seconds and 
            converted to number of elements of length <res> seconds
        :type seconds: bool
        :param res: Only relevant if <seconds> - length of each timestep in sequence
        :type res: float
        :param cache: If True, store a local cache of this matrix profile in <cache_dir>
        :type cache: bool
        :param cache_dir: Only relevant if <cache>, mp will be cahced locally to 
        :type cache_dir: bool
        
        :return: self matrix profile
        :rtype: numpy.array
        """

        return dmatrix

    def __unpack_json(self, j):
        """Unpack json, <j> to class attributes"""
        self.raaga = j.get('raaga', None)
        if self.raaga:
            self.raaga = self.raaga[0].get('name', None)

        self.taala = j.get('taala', None)
        if self.taala:
            self.taala = self.taala[0].get('name', None)

        self.album_artists = j.get('album_artists', None)
        if self.album_artists:
            self.album_artists = self.album_artists[0].get('name', None)

        self.concert = j.get('concert', None)
        if self.concert:
            self.concert = self.concert[0].get('title', None)

        self.title = j.get('title', None)
        self.length = j.get('length', None)

    def __get_svaras(self, raaga, svara_lookup_path):
        """
        Return arohana and avorahana for <raaga>
        from yaml at <svara_lookup_path>
        """
        lookup = load_yaml(svara_lookup_path)
        r_lookup = lookup[raaga]
        return r_lookup['arohana'], r_lookup['avorahana']


    def __get_svara_cents(self, svaras, svara_cent_path):
        """
        For iterable of <svaras> return dictionary of {svara: cents above tonic}
        """
        svara_cents = load_yaml(svara_cent_path)
        return {s:svara_cents[s] for s in svaras}
    

    def __mp_exists(self, dirname, file):
        path = os.path.join(dirname, file)
        if os.path.exists(path):
            return load_matrix_profile(path)
        else:
            return None

    def __getattr__(self, item):
        try:
            return getattr(self, item)
        except:
            return None

def compute_distance_matrix(rec, filt_perc, m_secs, nonzero_perc, filepath):
    create_if_not_exists(filepath)

    mp = rec.self_matrix_profile(m=m_secs, cache=True)
    MP = mp[:,0]

    mpfilt = MP[MP != np.Inf]
    mpfilt = MP[np.where(rec.pitch != 0)]

    thresh = np.percentile(mpfilt, filt_perc*100)

    m = int(m_secs/0.0029)
    T = rec.pitch
    lT = len(T)

    for i1 in tqdm.tqdm(range(int(lT))):
        
        if i1 > lT - m:
            break

        q = T[i1:i1+m]
        T_ = T[i1:]

        # Skip subsequences that are more than <nonzero_perc>% silence
        if not np.count_nonzero(q)/len(q) > nonzero_perc:
            continue
        
        stmass = stumpy.core.mass(q, T_)
        mass_below = stmass < thresh
        
        # Add i1 to convert to indice in T rather than in T_
        i2x = np.array([x+i1 for x in np.where(mass_below)[0]])
        i1x = np.array([i1]*len(i2x))
        
        dists = np.dstack([i1x, i2x, stmass[mass_below]])[0]

        with open(filepath, 'a') as f:
            np.savetxt(f, dists, delimiter=",")


def cluster_dist_matrix(dist_data, eps, intra_eps, min_n_clust):

    all_indices = set(dist_data[:,0]).union(set(dist_data[:,1]))
    n_ind = len(all_indices)
    index_seq = {int(i):int(s) for i,s in enumerate(all_indices)}
    seq_index = {s:i for i,s in index_seq.items()}

    index_lookup = dict(enumerate(all_indices))
    reverse_index_lookup = {v:k for k,v in index_lookup.items()}
    
    dist_data[:, 0] = [seq_index[int(i)] for i in dist_data[:,0]]
    dist_data[:, 1] = [seq_index[int(i)] for i in dist_data[:,1]]

    sparse_dist = csr_matrix(
        (dist_data[:,2], (dist_data[:,0].astype(int), dist_data[:,1].astype(int))), 
        shape=(n_ind, n_ind)
    )

    clustering = DBSCAN(eps=eps, metric='precomputed')
    clustering.fit(sparse_dist)

    labels = [(index_seq[i],l) for i,l in enumerate(clustering.labels_)]

    out_seq = []

    for c in set(clustering.labels_):
        if c == -1:
            continue
        cluster = np.array([x[0] for x in labels if x[1]==c])
        # Group sequences in the same neighbourhood (within <intra_eps> timesteps of each other)
        intra_clustering = DBSCAN(eps=intra_eps, metric='euclidean')
        intra_clustering.fit(cluster.reshape(-1,1))
        intra_labels = zip(cluster, intra_clustering.labels_)
        it = itertools.groupby(intra_labels, operator.itemgetter(1))
        cluster_ss = []
        for k, si in it:
            cluster_ss.append(min(si)[0])

        if len(cluster_ss) >= min_n_clust:
            out_seq.append(cluster_ss)
    
    return out_seq   

#performance_path = '/Users/thomasnuttall/mir_datasets/saraga_carnatic/saraga1.5_carnatic/Rithvik Raja at Arkay by Rithvik Raja/Chintayama Kanda/'

#test_rec = Recording(performance_path)
#test_rec.svara_cents
#mp = test_rec.self_matrix_profile(m=1000, cache=True)


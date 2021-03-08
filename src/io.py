import csv
import json 
import os
import yaml

#import essentia
#import essentia.standard
import numpy as np
import librosa
import soundfile as sf

from src.utils import get_timestamp

def audio_loader(path, sampleRate=44100):
    """
    Load audio file from <path> to numpy array

    :param path: Path of audio file compatiuble with essentia
    :type path: str
    :param sampleRate: sample rate of audio at <path>, default 44100
    :type sampleRate: int

    :return: Array of waveform values for each timestep
    :rtype: numpy.array
    """
    loader = essentia.standard.MonoLoader(filename = path, sampleRate = sampleRate)
    audio = loader()
    return audio


def write_pitch_contour(pitch, time, path):
    """
    Write pitch contour to tsv at <path>

    :param time: Array of time values for pitch contour
    :type time: numpy.array
    :param pitch: Array of corresponding pitch values
    :type pitch: numpy.array
    :param path: path to write pitch contour to
    :type path: str
    """
    ##text=List of strings to be written to file
    with open(path,'w') as file:
        for t, p in zip(time, pitch):
            file.write(f"{t}\t{p}")
            file.write('\n')


def load_pitch_contour(path):
    """
    load pitch contour from tsv at <path>

    :param path: path to load pitch contour from
    :type path: str

    :return: Two numpy arrays of time and pitch values
    :rtype: tuple(numpy.array, numpy.array)
    """
    time = []
    pitch = []
    with open(path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for t,p in rd:
            time.append(t)
            pitch.append(p)
    return np.array(time).astype(float), np.array(pitch).astype(float)


def load_json(path):
    """
    Load json at <path> to dict
    
    :param path: path of json
    :type path: str

    :return: dict of json information
    :rtype: dict
    """ 
    # Opening JSON file 
    with open(path) as f: 
        data = json.load(f) 
    return data


def load_yaml(path):
    """
    Load yaml at <path> to dictionary, d
    
    Returns
    =======
    Wrapper dictionary, D where
    D = {filename: d}
    """
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)   
    return d

def write_matrix_profile(matrix_profile, path):
    """
    Write matrix profile to tsv at <path>

    :param matrix_profile: from stumpy.stump, (matrix profile value, matrix profile index, left index, right index)
    :type matrix_profile: numpy.array
    :param path: path to write matrix profile to
    :type path: str
    """
    with open(path, 'w') as file:
        for mp, mpi, li, ri in matrix_profile:
            file.write(f"{mp}\t{mpi}\t{li}\t{ri}")
            file.write('\n')


def load_matrix_profile(path):
    """
    load pitch contour from tsv at <path>

    :param path: path to load pitch contour from
    :type path: str

    :return: Two numpy arrays of time and pitch values
    :rtype: tuple(numpy.array, numpy.array)
    """
    arr = []
    with open(path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for mp, mpi, li, ri in rd:
            arr.append(np.array([float(mp), int(mpi), int(li), int(ri)]))
    arr = np.array(arr, dtype=object)
    return arr


def load_tonic(path):
    """
    load tonic value frojm text file at <path>
    Text file should contain only a pitch number.

    :param path: path to load tonic from
    :type path: str

    :return: tonic in Hz
    :rtype: float
    """
    with open(path) as f:
        rd = f.read()
    return float(rd)

def create_if_not_exists(path):
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ''):
        os.makedirs(directory)


def write_subsequences_group(y, sr, starts, m_secs, output_dir):
    create_if_not_exists(output_dir)
    for s in starts:
        sec_start = s*0.0029        
        timestamp = get_timestamp(sec_start)
        out_path = os.path.join(output_dir, f'time={timestamp}.wav')
        l = m_secs*sr
        s1 = sec_start*sr
        s2 = s1+l
        subseq = y[int(s1):int(s2)]
        sf.write(out_path, subseq, samplerate=sr)

def write_all_sequence_audio(audio_path, all_seqs, m_secs, output_dir):
    y, sr = librosa.load(audio_path)
    for i, seq in enumerate(all_seqs):
        out_dir = os.path.join(output_dir, f'motif_{i}/')
        write_subsequences_group(y, sr, seq, m_secs, out_dir)

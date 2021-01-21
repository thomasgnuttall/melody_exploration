import essentia

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


def pitch_contour_writer(pitch, time, path):
    """
    Load audio file from <path> to numpy array

    :param time: Array of time values for pitch contour
    :type time: numpy.array
    :param pitch: Array of corresponding pitch values
    :type pitch: numpy.array
    :param path: path to write pitch contour to
    :type path: str
    """
    pass

######################
## Pitch Extractors ##
######################
# Pitch extractors should take as an input a numpy array representation of audio (+ kwargs)
#   ...and return two iterables; time and pitch, of equal length indicating the extracted pitch and given time
#

from essentia.standard import PredominantPitchMelodia
import numpy as np

def melodia(audio, frameSize, hopSize, sampleRate=44100.0):
    """
    Apply the Melodia algorithm to input audio to extract
    predominant pitch

    :param audio: [vector_real] audio signal
    :type audio: iterable
    :param frameSize: frame size for Fourier Transform
    :type frameSize: int
    :param hopSize: hop size for Fourier Transform
    :type hopSize: int
    :param sampleRate: sample rate of <audio>, default 441000.0
    :type sampleRate: int

    :return: tuple (time, pitch)
        time - np.array of time value
        pitch - np.array of extracted pitch values
    """

    # PitchMelodia takes the entire audio signal as input
    #   - no frame-wise processing is required here...
    pExt = PredominantPitchMelodia(frameSize = frameSize, hopSize = hopSize)
    pitch, pitchConf = pExt(audio)
    time = np.linspace(0.0, len(audio)/sampleRate, len(pitch))
    
    return time, pitch

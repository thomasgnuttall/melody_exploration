import matplotlib.pyplot as plt
import matplotlib.style as style
import scipy

style.use('seaborn-dark-palette')

iam_kwargs = {
    'window': scipy.signal.get_window('hann', int(0.0464*sampleRate)),
    'NFFT': int(0.0464*sampleRate), # window length x sample rate
}

def spectrogram(audio, sampleRate=44100, ylim=None, kwargs=iam_kwargs):
    """
    Plot spectrogram of input audio single

    :param audio: [vector_real] audio signal
    :type audio: iterable
    :param sampleRate: frame size for Fourier Transform
    :type sampleRate: int
    :param ylim: [min,max] frequency limits, default [20, 20000]
    :type ylim: iterable
    :param kwargs: Dict of keyword arguments for matplotlib.pyplot.specgram, default {}
    :type kwargs: dict

    :return:  spectrogram/waveform plot
    :rtype: matplotlib.pyplot
    """
    plt.title('Spectrogram')
    plt.specgram(audio, Fs=sampleRate, **kwargs)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.ylim(ylim)

    return plt
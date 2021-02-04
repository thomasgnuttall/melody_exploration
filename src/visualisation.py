from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import scipy.signal

style.use('seaborn-dark-palette')

iam_kwargs = {
    'window': scipy.signal.get_window('hann', int(0.0464*44100)),
    'NFFT': int(0.0464*44100), # window length x sample rate
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


def double_timeseries(x, y1, y2, y1label='', y2label='', xlabel='', xfreq=5, yfreq=50, linewidth=1):
    """
    Plot and two time series (with the same x) on the same axis

    :param x: x data (e.g time)
    :type array: numpy.array
    :param y1: y data of top plot
    :type y1: numpy.array
    :param y2: y data of bottom plot
    :type y2: numpy.array
    :param y1label: y label of top plot (y1 data)
    :type y1label: str
    :param y2label: y label of bottom plot (y2 data)
    :type y2label: str
    :param xfreq: y tick frequency
    :type xfreq: int
    :param yfreq: y tick frequency
    :type yfreq: int
    
    :return: tuple of plot objects, (fig, np.array([ax1, ax2]))
    :rtype: (matplotlib.figure.Figure, numpy.array([matplotlib.axes._subplots.AxesSubplot, ...]))
    """
    l = len(x)
    samp_len = int(l*0.0029)

    fig, axs = plt.subplots(2, sharex=True)
    fig.set_size_inches(170*samp_len/540, 10.5)
    axs[0].plot(x, y1, linewidth=linewidth)
    axs[0].grid()
    axs[0].set_ylabel(y1label)
    axs[0].set_xticks(np.arange(min(x), max(x)+1, xfreq))
    axs[0].set_yticks(np.arange(min(y1), max(y1)+1, yfreq))

    axs[1].plot(x[:len(y2)], y2, color='green', linewidth=linewidth)
    axs[1].grid()
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(y2label)
    axs[1].set_xticks(np.arange(min(x[:len(y2)]), max(x[:len(y2)])+1, xfreq))

    return fig, axs


def plot_annotate_save(x, y, matrix_profile, seqs, m, path, y1label='', y2label='', xlabel='Time'):
    """
    Plot and annotate time series and matrix_profile with 
    subsequences of length <m>, save png to <path>

    :param x: x data (e.g time)
    :type array: numpy.array
    :param y: y data (e.g. pitch) 
    :type y: numpy.array
    :param matrix_profile: Matrix profile of data in x,y
    :type matrix_profile: numpy.array
    :param seqs: List of subsequence start points to annotate on plot
    :type seqs: iterable
    :param m: Fixed length of subsequences in <seq>
    :type m: int
    :param path: Path to save png plot to
    :type path: str
    :param y1label: y label for time series
    :type y1label: str
    :param y2label: y label for matrix profile (distance measure)
    :type y2label: str
    :param xlabel: x label for time series, default 'Time'
    :type xlabel: str
    """
    fig, axs = double_timeseries(x, y, matrix_profile, y1label, y2label, xlabel)
    axs = annotate_plot(axs, seqs, m, linewidth=2)
    plt.savefig(path)
    plt.close('all')


def annotate_plot(axs, seqs, m, linewidth=2):
    """
    Annotate time series and matrix profile with sequences in <seqs>

    :param axs: list of two subplots, time series and matrix_profile
    :type axs: [matplotlib.axes._subplots.AxesSubplot, matplotlib.axes._subplots.AxesSubplot]
    :param seqs: iterable of subsequence start points to annotate
    :type seqs: numpy.array
    :param m: Fixed length of subsequences
    :type m: int
    :param linewidth: linewidth of shaded area of plot, default 2
    :type linewidth: float
    
    :return: list of two subplots, time series and matrix_profile, annotated
    :rtype: [matplotlib.axes._subplots.AxesSubplot, matplotlib.axes._subplots.AxesSubplot]
    """
    x_d = axs[0].lines[0].get_xdata()
    y_d = axs[0].lines[0].get_ydata()

    for c in seqs:
        x = x_d[c:c+m]
        y = y_d[c:c+m]
        axs[0].plot(x, y, linewidth=linewidth, color='burlywood')
        axs[1].axvline(x=x_d[c], linestyle="dashed", color='red')

    max_y = axs[0].get_ylim()[1]

    for c in seqs:
        rect = Rectangle((x_d[c], 0), m*0.0029, max_y, facecolor='lightgrey')
        axs[0].add_patch(rect)

    return axs

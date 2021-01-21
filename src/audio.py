import sounddevice as sd

def play_audio(audio, sampleRate=44100.0):
    """
    Play audio signal

    :param audio: [vector_real] audio signal
    :type audio: iterable
    :param sampleRate: frame size for Fourier Transform
    :type sampleRate: int
    """
    sd.play(audio, sampleRate)


def stop_audio():
    """
    If a sound is playing from play_audio, stop it
    """
    sd.stop()
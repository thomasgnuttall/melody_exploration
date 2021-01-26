def sample_audio(audio, f1, f2, sampleRate=44100):
	"""
	Sample <audio> between f1, f2

    :param audio: Array of waveform values for each timestep
    :type path: numpy.array
    :param f1: Beginning of desired sample in seconds
	:type f1: int
	:param f2: End of desired sample in seconds
	:type f2: int
    :param sampleRate: sample rate of audio at <path>, default 44100
    :type sampleRate: int

    :return: Array of waveform values for each timestep
    :rtype: numpy.array
	"""
	return audio[sampleRate*f1:sampleRate*f2]


def equal_loudness(audio):
	return audio

def source_separation(audio):
	return audio
%load_ext autoreload
%autoreload 2

path = '/Users/thomasnuttall/mir_datasets/saraga_carnatic/saraga1.5_carnatic/Chaitra Sairam at Arkay by Chaitra Sairam/Gange Maampahi/Chaitra Sairam - Gange Maampahi.mp3'

# load
from src.io import audio_loader

audio = audio_loader(path)

sample = audio[int(44100*3):int(44100*10)]


# Play
from src.audio import play_audio
play_audio(sample)


# Visualise
from src.visualisation import spectrogram

spec = spectrogram(sample)
spec.show()

# Extract pitch
from src.pitch_extractors import melodia 

time, pitch = melodia(sample, frameSize=2048, hopSize=128, sampleRate=44100)


# Combine
spec = spectrogram(sample)
spec.plot(time, pitch)
spec.show()
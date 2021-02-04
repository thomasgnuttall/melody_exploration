%load_ext autoreload
%autoreload 2

############################
# PITCH CONTOUR EXTRACTION #
############################

from src.io import audio_loader
from src.audio import play_audio
from src.visualisation import spectrogram
from src.pitch_extractors import melodia 

path = '/Users/thomasnuttall/mir_datasets/saraga_carnatic/saraga1.5_carnatic/Chaitra Sairam at Arkay by Chaitra Sairam/Gange Maampahi/Chaitra Sairam - Gange Maampahi.mp3'

# load
audio = audio_loader(path)
sample = audio[int(44100*3):int(44100*10)]

# Play
play_audio(sample)

# Visualise
spec = spectrogram(sample)
spec.show()

# Extract pitch
time, pitch = melodia(sample, frameSize=2048, hopSize=128, sampleRate=44100)

# Combine
spec = spectrogram(sample)
spec.plot(time, pitch)
spec.show()

############################
############################
############################


#######################
# MELODIC EXPLORATION #
#######################

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-dark-palette')

import numpy as np
import stumpy 

from src.chains import get_chains, average_chain, kde_cluster_1d, get_longest
from src.io import load_pitch_contour, load_json
from src.utils import get_stationary_point, find_nearest
from src.visualisation import double_timeseries, plot_annotate_save, annotate_plot

pitch_path = '/Users/thomasnuttall/mir_datasets/saraga_carnatic/saraga1.5_carnatic/V Shankarnarayanan at Arkay by V Shankaranarayanan/Ardhanareeshwaram/Ardhanareeshwaram.pitch-vocal.txt'

samp_len = 200 # in seconds
s_len = int(samp_len/0.0029) # in number elements

times, pitch = load_pitch_contour(pitch_path)
times = times[:s_len]
pitch = pitch[:s_len]

y1label = 'Pitch (Hz)'
y2label = 'Z-normalized Euclidean Distance'
xlabel = 'Time (seconds)'
plot_path = 'plots/MP_and_pitch.png'

# MATRIX PROFILE
################
# Pattern length in seconds 
pat_len = 2
m = int(pat_len/0.0029)
matrix_profile = stumpy.stump(pitch, m=m)
# returns array: [matrix profile, matrix profile indices, left matrix profile indices, right matrix profile indices]
MP = matrix_profile[:, 0]
MPI = matrix_profile[:, 1]



# GETTING SEQUENCE 
##################

# Non silent time steps whos suceeding <pat_len> seconds are also non-silent
# Array of 1s and 0s (non-silence or silence)
non_silence = np.array([0 if 0 in pitch[i:i+int(pat_len/0.0029)] else 1 for i in range(len(pitch))])
non_silence_index = np.array([x for x in range(len(non_silence)) if non_silence[x]])

orig_seq = get_stationary_point(MP, n_best=20, mask_mx=non_silence_index)

orig_mp_value = MP[orig_seq]
orig_time = times[orig_seq]

neighbour_seq = MPI[orig_seq]
neighbour_time = times[MPI[orig_seq]]

print(f'Subsequence loc (s): {round(orig_time,2)}')
print(f'Nearest Neighbour loc (s): {round(neighbour_time, 2)}')
print(f'MP Value: {round(orig_mp_value,2)}')


# PLOTTING
##########
seqs = [orig_seq, neighbour_seq]
plot_annotate_save(times, pitch, MP, seqs, m, plot_path)


# ALL CHAINS SET
################
all_chain_set = get_chains(matrix_profile[:,2], matrix_profile[:,3], mask=non_silence, m=m)
plot_annotate_save(times, pitch, MP, all_chain_set[0], m, plot_path)


all_chain_set_av = average_chain(all_chain_set)

#acs_val = [x[0] for x in all_chain_set_av]
#acs_ind = [x[1] for x in all_chain_set_av]

#plt.scatter(acs_val, range(len(acs_val)), s=1)
#plt.grid()
#plt.xlabel('Average initial timestep of sequences in chain')
#plt.ylabel('Chain #')
#plt.savefig('plots/chain_averages.png')
#plt.close('all')

# Cluster chains
bandwidth = m/av_chain_len
cluster_indices = kde_cluster_1d(all_chain_set_av, bandwidth=bandwidth)
all_chains_clustered = np.array(list(zip(all_chain_set, cluster_indices)), dtype=object)

# pick candidate
cluster_candidates = get_longest(all_chains_clustered)
# Plot some non silence chains
plot_annotate_save(times, pitch, MP, cluster_candidates[6], m, plot_path)

# DENSITY ESTIMATION
#################### 
#plt.plot(
#    s,
#    e,
#    s[ma], e[ma], 'go')
#plt.grid()
#plt.xlabel('Average initial timestep of sequences in chain')
#plt.ylabel('Kernel Density')
#plt.savefig('plots/chain_kde_estimation.png')
#plt.close('all')


# METADATA MATCH
################
metadata_path = '/Users/thomasnuttall/mir_datasets/saraga_carnatic/saraga1.5_carnatic/V Shankarnarayanan at Arkay by V Shankaranarayanan/Ardhanareeshwaram/Ardhanareeshwaram.json'
metadata = load_json(metadata_path)
raga = metadata['raaga'][0]['name']

# Source for south Indian info...
#   http://www.ibiblio.org/guruguha//ssp_7to12.pdf
# Sangita Sampradaya Pradarsini, an important south indian musicological text
# contains short raga descriptions

# Raga = Kumudakriyā (Janya of Kasiramakriya)
# Sargam = ra gu mi pa dha nu














# TODO
######

# area identifier
# get peak
# get trough
# get peak/trough within limits
#   return values and location
# annotate plot with box and colours
# changing distance function
    # distance function bespoke to indian music?
# play audio for given subsequence (requires essentia)



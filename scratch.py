#######################
# MELODIC EXPLORATION #
#######################
%load_ext autoreload
%autoreload 2

import csv
from collections import OrderedDict
import os

import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-dark-palette')

import numpy as np
import stumpy 

from src.chains import (
    get_chains, average_chain, kde_cluster_1d, get_longest, get_clustered_chains)
from src.io import (
    load_pitch_contour, load_json, write_matrix_profile, load_matrix_profile, 
    load_tonic)
from src.pitch.transformation import (
    octavize_pitch, locate_pitch_peaks, pitch_dict_octave_extend, 
    pitch_seq_to_cents, pitch_to_cents)
from src.pitch.carnatic import (
    get_shrutis)
from src.utils import (
    get_stationary_point, find_nearest, convert, myround, get_keyword_path)
from src.visualisation import (
    double_timeseries, plot_annotate_save, annotate_plot, double_timeseries,
    get_histogram, plot_pitch)

########################
### Load Performance ###
########################
# Raga Bhairavi
performance_path = '/Users/thomasnuttall/mir_datasets/saraga_carnatic/saraga1.5_carnatic/Rithvik Raja at Arkay by Rithvik Raja/Chintayama Kanda/'

arohana = ['S', 'G2', 'R2', 'G2', 'M1', 'P', 'D2', 'N2']
avarohana = ['S', 'N2', 'D1', 'P', 'M1', 'G2', 'R2']

pitch_path = get_keyword_path(performance_path, 'pitch-vocal')[0]
tonic_path = get_keyword_path(performance_path, 'tonic')[0]

freq_ratios = OrderedDict()
with open('data/svara_freq_ratios.tsv') as fd:
    sr = csv.reader(fd, delimiter="\t", quotechar='"')
    for s,r in sr:
        freq_ratios[s] = r

freq_ranges = OrderedDict()
with open('data/svara_freq_range.tsv') as fd:
    sr = csv.reader(fd, delimiter="\t", quotechar='"')
    for s, l, h in sr:
        freq_ranges[s] = [l,h]

times, pitch = load_pitch_contour(pitch_path)
tonic = load_tonic(tonic_path)

shrutis = get_shrutis(arohana, avarohana, return_svaras=False)
svaras = get_shrutis(arohana, avarohana, return_svaras=True)

# Shrutis or Svaras?
expected_freq = {f:[tonic*convert(a) for a in freq_ranges[f]]  for f in svaras}
#expected_freq = {f:tonic*convert(freq_ratios[f]) for f in shrutis}


########################
# Get pitch distribution
########################
# In Carnatic
upper_bound = 2*tonic + 10
lower_bound = tonic - 10
n_bins = 150

octavized_pitch = octavize_pitch(pitch, upper_bound, lower_bound)
vals, bins, bars = get_histogram(octavized_pitch, n_bins)
peaks_dict = locate_pitch_peaks(bins, vals, expected_freq)

# Force tonic since we know it
peaks_dict['S'] = tonic

peaks_dict_ext = pitch_dict_octave_extend(
    peaks_dict, lower_prefix='v', higher_prefix='^')

for name, peak in peaks_dict.items():
    plt.axvline(peak, color='red', linestyle='--', linewidth=0.5)
    ymax = plt.gca().get_ylim()[1]
    plt.annotate(name, (peak-1, ymax), color='red', fontsize=7)

plt.savefig('plots/pitch_histogram.png')
plt.close('all')
plt.cla()


###################################
# PLOT PITCH AS CENTS ABOVE TONIC #
################e###################
# Notes that are highlighted in red 
# e.g. those corresponding to multiple integers of the tonic
emphasis = ['Sv', 'S', 'S^']
none_mask = pitch == 0
samp_len = 10 # in seconds
s_len = (times > samp_len).argmax()

fig, ax = plot_pitch(
    pitch, times, s_len=s_len, mask=none_mask, yticks_dict=peaks_dict_ext,
    cents=True, tonic=tonic, emphasize=emphasis, figsize=(10,3))
plt.savefig('plots/chintayama_kanda/first_10.png', dpi=90)
plt.close('all')    
plt.cla()


################################
# Hysteresis for Svara Contour #
################################
# Ref - J. Canny. A computational approach to edge detection. [Hysteresis]

from scipy.ndimage import gaussian_filter1d
import pandas as pd

def get_notes_from_pitch(pitch, notes_dict):
    """
    For p in <pitch> assign a note value from keys
    in <notes_dict>
    """
    notes = list(notes_dict.keys())
    freq = list(notes_dict.values())
    pitch_notes = []
    for p in pitch:
        idx = find_nearest(freq, p, index=True)
        pitch_notes.append(notes[idx])
    return np.array(pitch_notes)


def rolling_average(arr, window):
    """
    Compute rolling average for every element in <arr>
    with a window of <window>. The first element of the array
    is returned unchanged 

    :param arr: Array
    :type arr: np.array
    :param window: Window size for average (in elements)
    :type window: int
    
    :return: Averaged array
    :rtype: np.array
    """
    a1 = np.convolve(arr, np.ones(window), 'valid') / window
    return np.insert(a1, 0, arr[:window-1], axis=0)

w_av = 0
s_ga = 200
sust_thresh = 0.2

pitch_fill = np.array([x if x else np.nan for x in pitch])
pitch_interp = pd.Series(pitch_fill)\
                 .interpolate(method='linear')\
                 .ffill()\
                 .bfill()\
                 .values

if s_ga:
    pitch_smoothed = gaussian_filter1d(pitch_interp, s_ga)
else:
    pitch_smoothed = pitch_interp
if w_av:
    pitch_average = rolling_average(pitch_smoothed, w_av)
else:
    pitch_average = pitch_smoothed 

pitch_notes = get_notes_from_pitch(pitch_average, peaks_dict_ext)
pitch_notes_time = list(zip(pitch_notes, times, none_mask))
just_notes_ = []
for i in range(len(pitch_notes_time[1:])):
    # Add onset if one of the following conditions are met
    conditions = [
        # If this points in sequence is not none and the last one was
        (
            (not pitch_notes_time[i-1][2]) and \
            (pitch_notes_time[i][2])
        ),
        # If the closest note at this point is different to the last (change of note)
        # and the element is not masked
        (
            (pitch_notes_time[i][0] != pitch_notes_time[i-1][0]) and \
            (not pitch_notes_time[i][2])
        )
    ]
    if any(conditions):
        just_notes_.append(pitch_notes_time[i])

# Remove notes that are not sustained for length longer than threshold
#just_notes = []
#t_ = np.NINF
#for i in range(len(just_notes_[1:])):
#    t = just_notes_[i][1]
#    if t - t_ > sust_thresh and just_notes_[-1][2]:
#        just_notes.append(just_notes_[-i])
#    t_ = t
just_notes = just_notes_

fig, ax = plot_pitch(
    pitch, times, s_len=s_len, mask=none_mask, yticks_dict=peaks_dict_ext,
    cents=True, tonic=tonic, emphasize=emphasis, title='Pitch Track with Automated Svara Annotations', ylim=(-800,1000))

max_t = max(times[:s_len])
for s,t, _ in just_notes:
    if t < max_t:
        ax.annotate(s, (t, 1000), color='darkgreen')
        ax.axvline(t, linestyle="dashed", color='darkgreen', linewidth=0.7)

plt.savefig('plots/svara_annotated_pitch.png', dpi=90)
plt.close('all')
plt.cla()




## From TANSEN : A SYSTEM FOR AUTOMATIC RAGA IDENTIFICATION
# Change in gradient of pitch contour is used for svara onset detection
    # sampling rate impacts a lot here
    # g_i1 = (p[i+1]-p[i])/(t[i+1]-t[i])
    # g_i = (p[i]-p[i-1])/(t[i]-t[i-1])
    # (g_1i-g_1)/g_1 > epsilon for new note
# A minimum note duration is hard coded (25ms)
# Curreent note is accepted as a note of the sample only if
# it is different from the dominant note in the history
    # ie the note which occurs more than m times in the history of k notes 
        # k=10, m=8






from skimage import data, filters

pitch_cents = pitch_seq_to_cents(pitch, tonic)
inf_pitch = np.array([x if not x is None else np.NINF for x in pitch_cents])


thresh = 50
high = pitch_seq_to_cents(peaks_dict['S'], tonic)
low = high - thresh
#high = v + u_thresh
#low = v - l_thresh
ht = filters.apply_hysteresis_threshold(inf_pitch, low, high)

fig, ax = plot_pitch(
    pitch, times, s_len=s_len, mask=none_mask, yticks_dict=peaks_dict_ext,
    cents=True, tonic=tonic, emphasize=emphasis)

binary_tonic = [-100 if all([x, not y]) else -800 for x,y in zip(ht,none_mask)]

ax.plot(times, binary_tonic)
plt.savefig('plots/hysteresis.png', dpi=90)
plt.close('all')
plt.cla()







def hysteresis(arr, v, tol_lo, tol_hi, initial = False):    
    th_lo = v - tol_lo
    th_hi = v + tol_hi
    
    hi = arr >= th_hi
    lo_or_hi = (arr <= th_lo) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size: # prevent index error if ind is empty
        return np.zeros_like(arr, dtype=bool) | initial
    cnt = np.cumsum(lo_or_hi) # from 0 to len(arr)
    return np.where(cnt, hi[ind[cnt-1]], initial)


peaks = sorted(peaks_dict_ext.items(), key=lambda y: -y[1])


hyst_dict = {}
for s,p in peaks:
    idx = hysteresis(inf_pitch, p, tol_lo=10, tol_hi=10)
    hyst_dict[s] = idx


hyst_out = []
for s,p in peaks_dict_ext.items():
    idx = hysteresis(inf_pitch, thresh=p, tol=30)
    hyst_out += list(zip(idx, [s]*len(idx)))

hyst_out = sorted(hyst_out)

hyst_dict = OrderedDict()

for i, s in hyst_out:
    hyst_dict.setdefault(i, []).append(s)


list(d.items())

# Re = [460, 600]
# Sa = [300,400,500]












# MATRIX PROFILE
################
import stumpy
import numpy as np
#from dask.distributed import Client

#dask_client = Client(n_workers=4)

y1label = 'Pitch (Hz)'
y2label = 'Z-normalized Euclidean Distance'
xlabel = 'Time (seconds)'
plot_path = 'plots/chintayama_kanda/best_sequence.png'

# Pattern length in seconds 
pat_len = 3
m = int(pat_len/0.0029)

matrix_profile = stumpy.stump(pitch, m=m)
write_matrix_profile(matrix_profile, 'data/Rithvik_Raja___Chintayama_Kanda/matrix_profile.tsv')

matrix_profile_smoothed = stumpy.stump(pitch_smoothed, m=m)
write_matrix_profile(matrix_profile_smoothed, 'data/Rithvik_Raja___Chintayama_Kanda/matrix_profile_smoothed.tsv')

matrix_profile = load_matrix_profile('data/Rithvik_Raja___Chintayama_Kanda/matrix_profile_smoothed.tsv')
MP = matrix_profile[:, 0]
MPI = matrix_profile[:, 1]


# Non silent time steps whos suceeding <pat_len> seconds are also non-silent
# Array of 1s and 0s (non-silence or silence)
non_silence = np.array([0 if 0 in pitch[i:i+int(pat_len/0.0029)] else 1 for i in range(len(pitch))])
non_silence_index = np.array([x for x in range(len(non_silence)) if non_silence[x]])

# 10.5 minutes normal
# 2 minutes withs scrump + pre scrump
# 9 seconds for scrump (perc=0.01) + update()
# less than a second for scrump (perc=0.01)

# Normal
# matrix_profile = stumpy.stump(pitch, m=m)
# returns array: 
#   [matrix profile, matrix profile indices, 
#       left matrix profile indices, right matrix profile indices]

# Scrump style
#scrump = stumpy.scrump(pitch, m=m, percentage=0.01, pre_scrump=False)
#scrump.update()
#MP = scrump.P_
#MPI = scrump.I_

fig, axs = double_timeseries(times[:s_len], pitch[:s_len], MP[:s_len], y1label, y2label, xlabel, ylim1=(0,600))
fig.savefig('chintayama_kanda.png')


# GETTING SEQUENCE 
##################
# top matches
top_n = 10000
best_sequence_ix = get_stationary_point(MP, top_n=top_n, mask_mx=non_silence_index)
best_sequences = [(i, MPI[int(i)]) for i in best_sequence_ix]


# PLOTTING
##########
i=-1
i+=1
plot_annotate_save(
    times, pitch, MP, best_sequences[i], 
    m, plot_path, y1label=y1label, y2label=y2label, 
    xlabel=xlabel, sample=True)



start_points = [longest]
pattern_length = 3

for i,sp in enumerate(start_points):
    this_pitch = pitch[sp:sp+int(pattern_length/0.0029)]
    this_times = times[sp:sp+int(pattern_length/0.0029)]
    this_mask = none_mask[sp:sp+int(pattern_length/0.0029)]
    fig, ax = plot_pitch(
        this_pitch, this_times, mask=this_mask, yticks_dict=peaks_dict_ext,
        cents=True, tonic=tonic, emphasize=emphasis, figsize=(10,4),
        xlim=(min(this_times), max(this_times)), ylim=(0, 1400))

    plt.savefig(f'plots/chintayama_kanda/motif_{i}.png', dpi=90)
    plt.close('all')    
    plt.cla()


distance_profile = enumerate(stumpy.core.mass(this_pitch, pitch))
distance_profile = sorted(distance_profile, key=lambda y: y[1])
    
sp = 62481
pattern_length = m
path = f'plots/chintayama_kanda/motif_2.png'
plot_kwargs = {
    'yticks_dict':peaks_dict_ext,
    'cents':True,
    'tonic':tonic, 
    'emphasize':emphasis, 
    'figsize':(10,4)#,
    #'ylim':(0, 1400)
}
MP_plot_kwargs = { 
    'figsize':(10,4),
    'ylabel':'Z-Normalized Euclidean Distance'
}
def plot_subsequence(sp, pattern_length, pitch, times, mask, path, plot_kwargs={}):
    this_pitch = pitch[sp:sp+int(pattern_length/0.0029)]
    this_times = times[sp:sp+int(pattern_length/0.0029)]
    this_mask = mask[sp:sp+int(pattern_length/0.0029)]
    fig, ax = plot_pitch(
        this_pitch, this_times, mask=this_mask,
        xlim=(min(this_times), max(this_times)), **plot_kwargs)

    plt.savefig(path, dpi=90)
    plt.close('all')    
    plt.cla()


plot_subsequence(sp, pattern_length, pitch, times, none_mask, path, plot_kwargs)

plot_subsequence(sp, 10, MP, times, none_mask, path, MP_plot_kwargs)

import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)[0], arr[detected_minima]

minimas, minima_values = detect_local_minima(MP)
min_sorted = sorted(list(zip(minima_values, minimas)))
min_sorted = [x for x in min_sorted if non_silence[x[1]]]

i = 3
ix = min_sorted[i][1]
plot_subsequence(ix, pattern_length, pitch, times, none_mask, 'plots/chintayama_kanda/test1.png', plot_kwargs)
plot_subsequence(MPI[ix], pattern_length, pitch, times, none_mask, 'plots/chintayama_kanda/test2.png', plot_kwargs)



# take (start points, euclid) and length
# group and return group leaders
# for all group leaders compute euclidean distance to all subsequences
# filter out thosee that overlap with current sequence (and any other matched sequences)
    # perhaps recursively apply step two (group and return leaders)
# cluster group leaders and matches 
# group metrics (intercluster distance, intracluster distance)
    # which are the strongest patterns?
# wrap in one line with m, caching, path, output of motif, audio, locations, certainty
# evalutation tbd

# ALL CHAINS SET
################
def get_timestamp(secs):
    """
    Convert seconds into timestamp

    :param secs: seconds
    :type secs: int

    :return: timestamp
    :rtype: str
    """
    minutes = int(secs/60)
    seconds = round(secs%60,4)
    return f'{minutes}min {seconds}sec'

all_chain_set = get_chains(matrix_profile[:,2], matrix_profile[:,3], mask=non_silence, m=m)

longest = sorted(all_chain_set, key=lambda y:-len(y))[0]
longest_sec = [get_timestamp(times[i]) for i in longest]

bandwidth = 1
all_chain_cluster_candidates = get_clustered_chains(
            all_chain_set, average_chain, kde_cluster_1d, 
            get_longest, cluster_func_kwargs={'bandwidth':bandwidth})



# Plot some non silence chains
i=-1
i+=1
plot_annotate_save(
    times, pitch, MP, longest, 
    m, plot_path, y1label=y1label, y2label=y2label, 
    xlabel=xlabel, sample=True)



cluster_candidates = get_clustered_chains(
            best_sequences, average_chain, kde_cluster_1d, 
            get_longest, cluster_func_kwargs={'bandwidth':bandwidth})

# Plot some non silence chains
i+=1
plot_annotate_save(
    times, pitch, MP, cluster_candidates[i], 
    m, plot_path, y1label=y1label, y2label=y2label, 
    xlabel=xlabel, sample=True)






# IDENTIFYING REGIONS
#####################
cac, regime_locations = stumpy.fluss(matrix_profile[:, 1], L=m, n_regimes=4, excl_factor=1)







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
# tonic identification
# apply steps from sankalp thesis
# svara distribution
# pitch normalised by tonic
# return patterns in peak numbers of svara distribution (or in svaras)
    # returning most significant sequences, longest chain
# Surpress Nyas
# normalise tempo
# Separate gamaka from pattern
# apply to larger sample of data
# area identifier
# get peak
# get trough
# get peak/trough within limits
#   return values and location
# annotate plot with box and colours
# changing distance function
    # distance function bespoke to indian music?
# play audio for given subsequence (requires essentia)



%load_ext autoreload
%autoreload 2

import mirdata

from src.core import run_pitch_extraction
from src.pitch_extractors import melodia
from src.preprocessing.melody_enhancer import melody_enhancer
from src.preprocessing.source_separation import source_separator
from src.validation.mirdata import validate_mirdata

#############
## Dataset ##
#############

carnatic = mirdata.initialize('saraga_carnatic')
carnatic.download()

hindustani = mirdata.initialize('saraga_hindustani')
hindustani.download()
hindustani_data = hindustani.load_tracks()

validate_mirdata(hindustani, verbose=False)
validate_mirdata(carnatic, verbose=False)

# To get for every mbid
# info_dict = {
#     'mbid1': {'pitch_track': None, 'metadata': None, 'pipeline_steps': None},
#     'mbid2': {'pitch_track': None, 'metadata': None, 'pipeline_steps': None},
#     'mbid3': {'pitch_track': None, 'metadata': None, 'pipeline_steps': None}
# }

carnatic_data = carnatic.load_tracks()
track_names = list(carnatic_data.keys())


# We have pitch tracks for "Cherthala Ranganatha Sharma at Arkay by Cherthala Ranganatha Sharma"
tracks_names_with_pitch = [x for x,y in carnatic_data.items() if y.concert[0]['title'] == 'Cherthala Ranganatha Sharma at Arkay']
ex_track_name = tracks_names_with_pitch[0]
ex_track = carnatic_data[ex_track_name]
in_path = ex_track.audio_vocal_path
out_path = '/Desktop/'











from src.preprocessing.standard import equal_loudness, source_separation, sample_audio
from src.pitch_extractors import melodia


conf = {
    # Load
    'audio_path': audio_path,
    'sampleRate': 44100.0,

    # Pre-processing
    'preprocessing_steps': [
            #(melody_enhancer, {}), 
            (equal_loudness, {}),
            (source_separation, {}),
            (sample_audio, {'f1':3, 'f2':10})
        ],
    
    # Predominant Pitch Extraction
    'extractor': melodia,
    'extractor_kwargs': {
                'frameSize': 2048, 
                'hopSize': 128, 
                'sampleRate': 44100
            },

    # add post-processing

    # Output
    'pitch_output_dir': f'/Users/thomasnuttall/Desktop/test_20210126.txt' # Doesn't write if None
}


run_pitch_extraction(conf)
























#########################
## Carnatic Extraction ##
#########################
for t in track_names:
    track = carnatic_data[t]
    if hasattr(track, 'audio_vocal_path'):
        audio_path = track.audio_vocal_path
        ap = 'vocal_track'
    elif hasattr(track, 'audio_path'):
        audio_path = track.audio_path
        ap='mixed_audio'
    else:
        continue
    conf = {
        # Load
        'audio_path': audio_path,
        'sampleRate': 44100.0,

        # Pre-processing
        'preprocessing_steps': [
                #(melody_enhancer, {}), 
                #(equal_loudness, {}),
                #(sampler, {})
            ],
        
        # Predominant Pitch Extraction
        'extractor': melodia,
        'extractor_kwargs': {
                    'frameSize': 2048, 
                    'hopSize': 128, 
                    'sampleRate': 44100
                },

        # add post-processing

        # Output
        'pitch_output_dir': f'/Users/thomasnuttall/Desktop/{t}___{ap}.txt' # Doesn't write if None
    }
    run_pitch_extraction(conf)





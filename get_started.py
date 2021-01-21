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

###############################
## Pitch Extraction Pipeline ##
###############################
conf = {
    # Load
    'audio_path': in_path,
    'sampleRate': 44100.0,

    # Pre-processing
    'preprocessing_steps': [
            (melody_enhancer, {}), 
            (source_separator, {})
        ],
    
    # Predominant Pitch Extraction
    'extractor': melodia,
    'extractor_kwargs': {
                'frameSize': 2048, 
                'hopSize': 128, 
                'sampleRate': 44100.0
            },

    # Output
    'pitch_output_dir': '/Users/thomasnuttall/Desktop/test.txt' # Doesn't write if None
}

time, pitch = run_pitch_extraction(conf)





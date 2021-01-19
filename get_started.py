%load_ext autoreload
%autoreload 2

import mirdata

from src.validation.mirdata import validate_mirdata

##############
## Get Data ##
##############

carnatic = mirdata.initialize('saraga_carnatic')
carnatic.download()

hindustani = mirdata.initialize('saraga_hindustani')
hindustani.download()
hindustani_data = hindustani.load_tracks()

validate_mirdata(hindustani, verbose=False)
validate_mirdata(carnatic, verbose=False)

###############
## Load Data ##
###############

# To get for every mbid
# info_dict = {
#     'mbid1': {'pitch_track': None, 'metadata': None, 'pipeline_steps': None},
#     'mbid2': {'pitch_track': None, 'metadata': None, 'pipeline_steps': None},
#     'mbid3': {'pitch_track': None, 'metadata': None, 'pipeline_steps': None}
# }

carnatic_data = carnatic.load_tracks() 
track = carnatic_data['248_Gange_Maampahi'] # docstring below*



######################################
## Pitch Extraction (if not exists) ##
######################################



######################
## Melodic Analysis ##
######################

# part a

# part b







#### *Track ####
##
##
##      Args:
##          track_id (str): track id of the track
##          data_home (str): Local path where the dataset is stored. default=None
##              If `None`, looks for the data in the default directory, `~/mir_datasets`
##      
##      Attributes:
##          title (str): Title of the piece in the track
##          mbid (str): MusicBrainz ID of the track
##          album_artists (list, dicts): list of dicts containing the album artists present in the track and its mbid
##          artists (list, dicts): list of dicts containing information of the featuring artists in the track
##          raaga (list, dict): list of dicts containing information about the raagas present in the track
##          form (list, dict): list of dicts containing information about the forms present in the track
##          work (list, dicts): list of dicts containing the work present in the piece, and its mbid
##          taala (list, dicts): list of dicts containing the talas present in the track and its uuid
##          concert (list, dicts): list of dicts containing the concert where the track is present and its mbid
##      
##      Cached Properties:
##          tonic (float): tonic annotation
##          pitch (F0Data): pitch annotation
##          pitch_vocal (F0Data): vocal pitch annotation
##          tempo (dict): tempo annotations
##          sama (BeatData): sama section annotations
##          sections (SectionData): track section annotations
##          phrases (SectionData): phrase annotations
##      
##      Method resolution order:
##          Track
##          mirdata.core.Track
##          builtins.object
##      
##      Methods defined here:
##      
##      __init__(self, track_id, data_home)
##          Initialize self.  See help(type(self)) for accurate signature.
##      
##      phrases = <mirdata.core.cached_property object>
##      pitch = <mirdata.core.cached_property object>
##      pitch_vocal = <mirdata.core.cached_property object>
##      sama = <mirdata.core.cached_property object>
##      sections = <mirdata.core.cached_property object>
##      tempo = <mirdata.core.cached_property object>
##      to_jams(self)
##          Get the track's data in jams format
##      
##          Returns:
##              jams.JAMS: the track's data in jams format
##      
##      tonic = <mirdata.core.cached_property object>


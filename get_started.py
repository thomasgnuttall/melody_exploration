%load_ext autoreload
%autoreload 2

import mirdata

from src.validation.mirdata import validate_mirdata

## Get Data ##

carnatic = mirdata.initialize('saraga_carnatic')
carnatic.download()

hindustani = mirdata.initialize('saraga_hindustani')
hindustani.download()

validate_mirdata(hindustani, verbose=False)
validate_mirdata(carnatic, verbose=False)


## Load Data ##


## Pitch Extraction (if not exists) ##


## Melodic Analysis ##

# part a

# part b

class DatasetException(Exception):
    pass


def validate_mirdata(dataset, verbose=False):
	"""
	Validate that
		Files in the index but are missing locally
		Files which have an invalid checksum

	:param dataset: locally available mirdata dataset
	:type dataset: mirdata.core.Dataset

	:param verbose: Print out tracks that fail validation? Defaults to False
	:type verbose: bool
	"""
	d1, d2 = dataset.validate()

	validate_mirdata_index(d1, verbose=verbose)
	validate_mirdata_checksum(d2, verbose=verbose)


def validate_mirdata_index(d, verbose=False):
	"""
	Validate that all files in the index but are available locally
	
	:param d: First returned dict from mirdata.core.Dataset validate() method
	:type d: dict

	:param verbose: Print out tracks that fail validation? Defaults to False
	:type verbose: bool
	"""
	tracks = d['tracks']
	if tracks:
		if verbose:
			m = (f"The following files in the index are"\
				  " not available locally: \n{tracks}")
		else:
			m = (f"There are files in the index"\
				  " that are not available locally")
		raise DatasetException(m)


def validate_mirdata_checksum(d, verbose=False):
	"""
	Validate that all files have a valid checksum
	
	:param d: Second returned dict from mirdata.core.Dataset validate() method
	:type dataset: dict

	:param verbose: Print out tracks that fail validation? Defaults to False	
	:type verbose: bool
	"""
	tracks = d['tracks']
	if tracks:
		if verbose:
			m = (f"The following files in the index do"\
				  " not have a valid checksum: \n{tracks}")
		else:
			m = (f"There are files in the index"\
				  " that do not have a valid checksum")
		raise DatasetException(m)
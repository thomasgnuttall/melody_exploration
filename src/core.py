from src.io import audio_loader, write_pitch_contour

def run_pitch_extraction(conf):
    """
    Run pitch extraction pipeline using parameters specified in <conf>

    :param conf: Dict of configuration parameters for pipeline
    :type conf: dict
    """
    # Load
    # TODO: replace with logger
    path = conf['audio_path']
    print(f'Loading audio from: {path}')
    audio = audio_loader(path, sampleRate=conf['sampleRate'])

    # pre process
    for pp_func, pp_kwargs in conf['preprocessing_steps']:
        print(f'Pre-processing step: {pp_func.__name__}')
        audio = pp_func(audio, **pp_kwargs)

    extractor = conf['extractor']
    extractor_kwargs = conf['extractor_kwargs']
    
    # Extract melody
    print(f'Extracting pitch with extractor: {extractor.__name__}')
    time, pitch = extractor(audio, **extractor_kwargs)

    out_path = conf['pitch_output_dir']
    if out_path:
        print(f'Writing pitch contour to {out_path}')
        write_pitch_contour(pitch, time, out_path)

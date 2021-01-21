from src.io import audio_loader, pitch_contour_writer

def run_pitch_extraction(conf):
    """
    Run pitch extraction pipeline using parameters specified in <conf>

    :param conf: Dict of configuration parameters for pipeline
    :type conf: dict
    """
    # Load
    # TODO: replace with logger
    path = conf['audio_path']
    print(f'Loading audio at {path}')
    audio = audio_loader(path, sampleRate=conf['sampleRate'])
    sample = audio[5*sampleRate:10*sampleRate]

    for pp_func, pp_kwargs in conf['preprocessing_steps']:
        print(f'Pre-processing step: {pp_func.__name__}')
        sample = pp_func(sample, **pp_kwargs)

    extractor = conf['extractor']
    extractor_kwargs = conf['extractor_kwargs']
    
    print(f'Extracting pitch with extractor: {extractor.__name__}')
    time, pitch = extractor(audio, **extractor_kwargs)

    out_path = conf['pitch_output_dir']
    if out_path:
        print('Writing pitch contour to {out_path}')
        src.io.pitch_contour_writer(pitch, time, out_path)

    return time, pitch
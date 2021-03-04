# All 16 shrutis in carnatic tradition
shrutis = ['S','R1','R2','G1','R3','G2','G3','M1','M2','P','D1','D2','N1','D3','N2','N3']
# All 7 svaras in carnatic tradition
svaras = ['S', 'R', 'G', 'M', 'P', 'D', 'N']

def get_shrutis(arohana, avarohana, return_svaras=False):
    """
    From input arohana and avarohana get a list of all unique shruti names in the raga

    :param arohana: list of strings of srutis in arohana ('S', 'R1', 'G2' etc...)
    :type arohana: list(str)
    :param avarohana: list of strings of srutis in avarohana ('S', 'R1', 'G2' etc...)
    :type avarohana: list(str)
    :param return_svaras: If True, only return the svara (without microtonal variation), default False
    :type return_svaras: bool

    :return: All srutis/svaras in raga in ascending pitch order
    :rtype: list
    """
    all_freqs = set(arohana).union(set(avarohana))
    all_svaras = set([''.join([f for f in freq if not f.isdigit()])\
                     for freq in all_freqs])
    if return_svaras:
        # maintain order returning like this
        return [x for x in svaras if x in all_svaras]

    relevant_shrutis = []
    for s in all_svaras:
        relevant_shrutis += [x for x in shrutis if s in x]
    
    # maintain order returning like this
    return [x for x in shrutis if x in set(relevant_shrutis)]

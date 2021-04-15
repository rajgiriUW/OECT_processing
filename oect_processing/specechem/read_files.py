import numpy as np
import os
import pandas as pd
import re
from pathlib import Path


def read_files(path):
    '''
    Takes a folder and finds the potential from all the "Steps" files
        
    Input
    -----
    path : str
        Folder path to where the data are contained. Assumes are saved as "steps"
    
    Returns
    -------
    stepfiles : list of strings
        For the "steps" (current)
    specfiles : list of string
        For the list of spectra files
    potentials : ndarray
        Numpy array of the potentials in filelist order
    '''
    if isinstance(path, str):
        path = Path(path)

    filelist = [f for f in os.listdir(path) if not f.startswith('.')]

    # Rename the first files
    if 'steps.txt' in filelist:
        os.rename(path / 'steps.txt', path / 'steps(0).txt')
    if 'spectra.txt' in filelist:
        os.rename(path / 'spectra.txt', path / 'spectra(0).txt')
    if 'stepsspectra.txt' in filelist:
        os.rename(path / 'stepsspectra.txt', path / 'stepsspectra(0).txt')
    if 'dedoping.txt' in filelist:
        os.rename(path / 'dedoping.txt', path / 'dedoping(0).txt')
    if 'dedopingspectra.txt' in filelist:
        os.rename(path / 'dedopingspectra.txt', path / 'dedopingspectra(0).txt')

    filelist = os.listdir(path)

    stepfiles = [os.path.join(path, name)
                 for name in filelist if (name[-3:] == 'txt' and 'steps(' in name)]
    specfiles = [os.path.join(path, name)
                 for name in filelist if (name[-3:] == 'txt' and 'spectra(' in name
                                          and 'dedoping' not in name)]
    dedopestepfiles = [os.path.join(path, name)
                       for name in filelist if (name[-3:] == 'txt' and 'dedoping(' in name)]
    dedopespecfiles = [os.path.join(path, name)
                       for name in filelist if (name[-3:] == 'txt' and 'dedopingspectra(' in name)]

    ''' Need to "human sort" the filenames or sorts 1,10,11,2,3,4, etc'''

    # https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
    def natural_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    specfiles = natural_sort(specfiles)
    stepfiles = natural_sort(stepfiles)
    dedopespecfiles = natural_sort(dedopespecfiles)
    dedopestepfiles = natural_sort(dedopestepfiles)

    potentials = np.zeros([len(stepfiles)])

    pp = pd.read_csv(stepfiles[0], header=0, sep='\t')
    pot = [n for n in pp.columns if 'Potential' in n][0]

    for fl, x in zip(stepfiles, np.arange(len(potentials))):
        pp = pd.read_csv(fl, header=0, sep='\t')
        potentials[x] = np.round(pp[pot][0], 2)

    return stepfiles, specfiles, potentials, dedopestepfiles, dedopespecfiles

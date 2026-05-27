import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_files(path):
    '''
    Finds and sorts all step and spectra files in a folder, and extracts potentials.

    Parameters
    ----------
    path : str or Path
        Folder containing the data files, saved as "steps" and "spectra".

    Returns
    -------
    stepfiles : list of str
        Sorted list of doping step (current) file paths.
    specfiles : list of str
        Sorted list of doping spectra file paths.
    potentials : ndarray
        Applied potentials in filelist order.
    dedopestepfiles : list of str
        Sorted list of dedoping step file paths.
    dedopespecfiles : list of str
        Sorted list of dedoping spectra file paths.
    '''
    if isinstance(path, str):
        path = Path(path)
    print(path)
    filelist = [str(f) for f in os.listdir(path) if not f.startswith('.')]

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

    # single pass to categorise all txt files
    stepfiles, specfiles, dedopestepfiles, dedopespecfiles = [], [], [], []
    for name in filelist:
        if not name.endswith('.txt'):
            continue
        full = os.path.join(path, name)
        if 'dedopingspectra(' in name:
            dedopespecfiles.append(full)
        elif 'dedoping(' in name:
            dedopestepfiles.append(full)
        elif 'spectra(' in name:
            specfiles.append(full)
        elif 'steps(' in name:
            stepfiles.append(full)

    # https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
    def natural_sort(l):
        def alphanum_key(key):
            return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    stepfiles = natural_sort(stepfiles)
    specfiles = natural_sort(specfiles)
    dedopestepfiles = natural_sort(dedopestepfiles)
    dedopespecfiles = natural_sort(dedopespecfiles)

    # detect potential column from first file only, then read one row per file
    first = pd.read_csv(stepfiles[0], header=0, sep='\t', nrows=1)
    try:
        pot = [n for n in first.columns if 'Potential' in n][0]
    except:
        pot = [n for n in first.columns if 'Vf' in n][0]

    potentials = np.zeros(len(stepfiles))
    potentials[0] = np.round(first[pot][0], 2)
    for x, fl in enumerate(stepfiles[1:], start=1):
        pp = pd.read_csv(fl, header=0, sep='\t', nrows=1)
        potentials[x] = np.round(pp[pot][0], 2)

    return stepfiles, specfiles, potentials, dedopestepfiles, dedopespecfiles

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:36:20 2018

@author: Raj
"""

import numpy as np
import pandas as pd
import os
import re

def read_files(path):
    '''
    Takes a folder and finds the potential from all the "Steps" files
    NOTE: rename "steps" and "Stepspectra" to "steps(0)" and "stepspectra(0)", respectively
    
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
    
    
    filelist = os.listdir(path)
    stepfiles = [os.path.join(path, name)
                 for name in filelist if (name[-3:] == 'txt' and 'spectra' not in name)]
    specfiles = [os.path.join(path, name)
                 for name in filelist if (name[-3:] == 'txt' and 'spectra' in name)]
    
    ''' Need to "human sort" the filenames '''
    # https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
    def natural_sort(l): 
        convert = lambda text: int(text) if text.isdigit() else text.lower() 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)

    specfiles = natural_sort(specfiles)
    stepfiles = natural_sort(stepfiles)
    
    potentials = np.zeros([len(stepfiles)])

    pp = pd.read_csv(stepfiles[0], header=0, sep='\t')
    pot = [n for n in pp.columns if 'Potential' in n][0]

    for fl, x in zip(stepfiles, np.arange(len(potentials))):
    
        pp = pd.read_csv(fl, header=0, sep='\t')
        potentials[x] = np.round(pp[pot][0],2)
        
    return stepfiles, specfiles, potentials

def spec_echem(specfiles, potentials, which_run=-1):
    '''
    Takes the list of spectra files specfiles, then extracts the final spectra
    from each file and returns as a single dataframe
    
    specfiles : list of str
        Contains paths to the specfiles
        
    potentials : list
        Contains correlated list of Gate voltages
        
    which_run : int, optional
        Which run to select and save. By default is the last (the final time slice)
        
    Returns
    -------
    df : pandas Dataframe
    
    '''
    pp = pd.read_csv(specfiles[0], sep='\t')
    
    runs = np.unique(pp['Spectrum number'])
    per_run = int(len(pp)/runs[-1])
    last_run = runs[-1]
    wl = pp['Wavelength (nm)'][0:per_run]
    
    # Set up dataframe
    df = pd.DataFrame(index=wl)

    for fl,v in zip(specfiles, potentials):
        
        pp = pd.read_csv(fl, sep='\t')
        data = pp[pp['Spectrum number']==last_run]['Absorbance'].values
        df[v] = pd.Series(data, index=df.index)
        
    
    return df

def single_time_spectra(spectra_path):
    '''
    Plots the time-dependent spectra for a single dataframe
    
    spectra_path : str
        Path to a specific spectra file
    '''
    
    pp = pd.read_csv(spectra_path, sep='\t')
    
    runs = np.unique(pp['Spectrum number'])
    times = np.unique(pp['Time (s)'])
    per_run = int(len(pp)/runs[-1])
    wl = pp['Wavelength (nm)'][0:per_run]
    
    # Set up dataframe
    df = pd.DataFrame(index=wl)
    
    for k, t in zip(runs, times):
        
        data = pp[pp['Spectrum number']==k]['Absorbance'].values
        df[np.round(t,2)] = pd.Series(data, index=df.index)
    
    return df
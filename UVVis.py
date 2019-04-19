# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:36:20 2018

@author: Raj
"""

import numpy as np
import pandas as pd
from scipy import signal as sg
from scipy.optimize import curve_fit
import os
import re

from matplotlib import pyplot as plt

'''
UV Vis spec-echem processing

Usage:
    
    >> steps, specs, potentials = UVVis.read_files(path_to_folder)
    >> data = UVVis.uv_vis(steps,specs,potentials)
    >> data.spec_echem_voltage() # spectra at each voltage in one dataFrame
    >> data.time_dep_spectra() # generates a dict of spectra vs time at a voltage
    >> data.single_wl_time(0.8, 800) # wavelength vs time at a given bias

'''
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
                 for name in filelist if (name[-3:] == 'txt' and 'steps(' in name)]
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


class uv_vis(object):
    
    def __init__(self, steps, specs, potentials):
        '''
        steps : list
            List of step(current) files
            
        specs : list
            List of spectra files
            
        potentials : list
            List of voltages applied
            
        Class contains:
        --------------
        self.steps
        self.specs
        self.potentials
        self.spectra : pandas Dataframe
            The spectra at each voltage in a large dataframe
        self.spectra_sm : pandas DataFrame
            The smoothed spectra at each voltage in a large dataFrame
        self.vt : pandas Series
            The voltage spectra at a particular wavelength (for threshold measurement)
        self.time_spectra_norm : pandas Series
            Normalized Time spectra at a given wavelength and potential over time
        self.time_spectra : pandas Series
            Time spectra at a given wavelength and potential over time
        '''
        self.steps = steps
        self.specs = specs
        self.potentials = potentials
        
        self.wavelength = 800
        
        return
    
    def spec_echem_voltage(self, wavelength=800, which_run=-1, smooth=3):
        '''
        Takes the list of spectra files specfiles, then extracts the final spectra
        from each file and returns as a single dataframe
        
        Also extracts the absorbance vs voltage at a particular wavelength
        
        specfiles : list of str
            Contains paths to the specfiles
            
        potentials : list
            Contains correlated list of Gate voltages
            
        wavelength : int
            wavelength to extract voltage-dependent data on
            
        which_run : int, optional
            Which run to select and save. By default is the last (the final time slice)

        smooth : int
            simple boxcar smooth of data for plotting/analysis purposes, controls
            size of filter 
            
        Saves:
        ------
        spectra : time=0 spectra at each voltage
        spectra_sm : time=0 spectra (smoothed) at each voltage
        vt : absorbance at 'wavelength' vs voltage (like a transfer curve)
            
        '''  
        pp = pd.read_csv(self.specs[0], sep='\t')
        
        runs = np.unique(pp['Spectrum number'])
        per_run = int(len(pp)/runs[-1])
    #    last_run = runs[-1]
        wl = pp['Wavelength (nm)'][0:per_run]
        
        # Set up dataFrame
        df = pd.DataFrame(index=wl)
        
        # Set up a "smoothed" dataFrame
        dfs = pd.DataFrame(index=wl)
    
        for fl,v in zip(self.specs, self.potentials):
            
            pp = pd.read_csv(fl, sep='\t')
            data = pp[pp['Spectrum number']==runs[which_run]]['Absorbance'].values
            df[v] = pd.Series(data, index=df.index)
            data = sg.fftconvolve(data, np.ones(smooth)/smooth, mode='same')
            dfs[v] = pd.Series(data, index=df.index)
    
        idx = df[v].index
        wl = idx.searchsorted(wavelength)
        self.vt = df.loc[idx[wl]]
        
        self.spectra = df
        self.spectra_sm = dfs
        self._single_wl_voltage(wavelength) # absorbance vs voltage @ a wavelength
        
        return 
    
    def time_dep_spectra(self, smooth=3):
        '''
        Generates all the time-dependent spectra
        '''
        
        self.spectra_vs_time = {}
        for v in self.potentials:
            
            spectra_path = self.specs[self.volt(v)]
            df = self.single_time_spectra(spectra_path)
            
            self.spectra_vs_time[v] = df
        
        return

    def single_wl_time(self, potential=0, wavelength=800, smooth=3):
        '''
        Extracts the time-dependent data from a single wavelength
        
        potential : float
            Find run corresponding to potential. Note in UV-Vis substrate is biased, not gate electrode
            
        wavelength : int, float
            Wavelength to extract. This will search for nearest wavelength row
            
        '''
        df = self.spectra_vs_time[potential]
        
        idx = df.index
        wl = idx.searchsorted(wavelength)
        
        data = df.loc[idx[wl]] - np.min(df.loc[idx[wl]])
        data =  data / np.max(data)
        
        self.time_spectra = df.loc[idx[wl]]
        self.time_spectra_norm = pd.Series(data.values, index=df.loc[idx[wl]].index)

        # smooth
        for c in df.columns:
            df[c] = sg.fftconvolve(df[c], np.ones(smooth)/smooth, mode='same')

        data = df.loc[idx[wl]] - np.min(df.loc[idx[wl]])
        data = data / np.max(data)
        
        self.time_spectra_sm = df.loc[idx[wl]]
        self.time_spectra_norm_sm = pd.Series(data.values, index=df.loc[idx[wl]].index)
        
        return 
    
    def _single_wl_voltage(self, wavelength=800):
        '''
        Extracts the absorbance vs voltage at a particular wavelength (threshold visualizing)
        '''
        idx = self.spectra.index
        wl = idx.searchsorted(wavelength)
        self.vt = self.spectra.loc[idx[wl]]
        
        return 
    
    def single_time_spectra(self,spectra_path):
        '''
        Generates the time-dependent spectra for a single dataframe
        
        spectra_path : str
            Path to a specific spectra file
        '''
        
        pp = pd.read_csv(spectra_path, sep='\t')
        
        runs = np.unique(pp['Spectrum number'])
        times = np.unique(pp['Time (s)'])
        times = times - times[0]
        per_run = int(len(pp)/runs[-1])
        wl = pp['Wavelength (nm)'][0:per_run]
        
        # Set up dataframe
        df = pd.DataFrame(index=wl)
        
        for k, t in zip(runs, times):
            
            data = pp[pp['Spectrum number']==k]['Absorbance'].values
            df[np.round(t,2)] = pd.Series(data, index=df.index)
        
        return df
    

    def volt(self,bias):
        '''
        returns voltage from potential list
        
        '''
        out = np.searchsorted(self.potentials, bias)
        
        return out

    def banded_plot(self, wl_start=700, wl_stop = 900, voltage=1, fittype='exp'):
        '''
        Returns the fits from a range of spectra_vs_time data
        '''
        
        wl_x = self.spectra_vs_time[voltage][wl_start:wl_stop]
        tx = self.time_spectra_norm_sm.index.values
        
        if fittype not in ['exp', 'biexp', 'stretched']:
            raise ValueError('Fit must be exp, biexp, or stretched')
         
        fits = [] #single exponential
        
        for wl in wl_x.index.values[1:]:
            
            if fittype == 'exp':
                popt, _ = curve_fit(fit_exp, tx, self.spectra_vs_time[voltage].loc[wl])
                fits.append(popt[2])
            elif fittype == 'biexp':
                popt, _ = curve_fit(fit_biexp, tx, self.spectra_vs_time[voltage].loc[wl])
                fits.append((popt[2], popt[4]))
            elif fittype == 'stretched':
                popt, _ = curve_fit(fit_strexp, tx, self.spectra_vs_time[voltage].loc[wl])
                fits.append((popt[2], popt[3]))



def fit_exp(t, y0, A, tau):
    
    return y0 + A * np.exp(-t/tau)

def fit_biexp(t, y0, A1, tau1, A2, tau2):
    
    return y0 + A1 * np.exp(-t/tau1) + A2 * np.exp(-t/tau2)

def fit_strexp(t, y0, A, tau, beta):
    
    return y0 + A * (np.exp(-t/tau))**beta


        
    
    return fits

   
def plot_time(uv, ax=None, norm=True, smooth=False, **kwargs):
    
    if ax == None:
        fig, ax = plt.subplots(nrows=1, figsize=(12, 6))
    
    if smooth:

        if norm:
            uv.time_spectra_norm_sm.plot(ax=ax, **kwargs)
        else:
            uv.time_spectra_sm.plot(ax=ax, **kwargs)
            
    else:
        
        if norm:
            uv.time_spectra_norm.plot(ax=ax, **kwargs)
        else:
            uv.time_spectra.plot(ax=ax, **kwargs)
        
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalize absorbance (a.u.)')
    
    return ax

def plot_spectra(uv, ax=None, smooth=False, **kwargs):
    
    if ax == None:
        fig, ax = plt.subplots(nrows=1, figsize=(12, 6))
        
    if smooth:
        uv.spectra_sm.plot(ax=ax, **kwargs)
    else:
        uv.spectra.plot(ax=ax, **kwargs)
        
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance (a.u.)')
    
    return ax

def plot_voltage(uv, ax=None, norm=None, **kwargs):
    '''
    norm = normalize the threshold UV-Vis data
    '''
    if ax == None:
        fig, ax = plt.subplots(nrows=1, figsize=(12, 6))
    
    if norm == None:
        ax.plot(uv.potentials*-1, uv.vt.values, **kwargs)
    else:
        numerator = (uv.vt.values-uv.vt.values.min())
        ax.plot(uv.potentials*-1, numerator/numerator.max(), **kwargs)
    ax.set_xlabel('Gate Bias (V)')
    ax.set_ylabel('Absorbance (a.u.)')
    
    return ax
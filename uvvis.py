# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:36:20 2018

@author: Raj
"""

import numpy as np
import pandas as pd
from scipy import signal as sg
from scipy.optimize import curve_fit
from scipy import integrate as spint
import os
import re
import h5py
from pathlib import Path

import seaborn as sns

from matplotlib import pyplot as plt

'''
UV Vis spec-echem processing

Usage:
    
    >> steps, specs, potentials,_,_ = uvvis.read_files(path_to_folder)
    >> data = uvvis.uv_vis(steps, specs, potentials)
    >> data.time_dep_spectra(specfiles=specs)  # Dict of spectra vs time
    >> data.single_wl_time(0.8, 800) # wavelength vs time at a given bias (0.8 V) and wavelength (800 nm)
    >> data.abs_voltage(800, 20) # absorbance vs voltage at specific wavelength (800 nm) and specific time (20 s)
    
    >> uvvis.plot_voltage(data)

'''


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


class uv_vis(object):

    def __init__(self, steps=None, specs=None, potentials=None):
        '''
        steps : list
            List of step(current) files
            
        specs : list
            List of spectra files
            
        potentials : list
            List of voltages applied
            
        Class contains:
        --------------
        spectra : pandas Dataframe
            The spectra at each voltage in a large dataframe
        spectra_sm : pandas DataFrame
            The smoothed spectra at each voltage in a large dataFrame
        spectra_vs_time : dict
            Dict of spectra vs time correspondign to each voltage.
            i.e. uv_viss.spectra_vs_time[1] is all the time-dependent spectra at 1 V
        current : pandas DataFrame
            The time-resolved current at each voltage step in a single dataFrame
        time_spectra : pandas Series
            Time spectra at a given wavelength and potential over time
        time_spectra_norm : pandas Series
            Normalized Time spectra at a given wavelength and potential over time
            
        vt : pandas Series
            The voltage spectra at a particular wavelength (for threshold measurement)
        tx : ndArray
            The time-axis used for plotting and fitting time
        fits : ndArray
            The fits generated from banded_fits (exponential fits to time-slices)
        '''
        self.steps = steps
        self.specs = specs
        self.potentials = potentials

        return

    def time_dep_spectra(self, specfiles, smooth=None, round_wl = 2, droptimes = None):
        '''
        Generates all the time-dependent spectra. This yields a dictionary where
        each voltage key contains the time-dependent spectra dataframe 
        
        e.g. at spectra_vs_time[0.9] you have the spectra for potential of 0.9 V
        in a dataFrame where wavelength is the index and each column is time-slice
        
        This dict is the major component of this Class
        
        specfiles : list of str
            Contains paths to the spectra files on the disk somewhere
        
        smooth : int, optional
            For smoothing the spectra via a boxcar filter. None = no smoothing
            Typical value is 3 (i.e. 3 point smoothing)
        
        round_wl : int, optional
            Digits to round the wavelength values to in the dataFrames
            None = no rounding
            
        droptimes : list or array, optional
            Specific time indices to drop. This functionality is for initial or final
            errors in spectroelectrochemistry data
        
        '''

        self.spectra_vs_time = {}
        for v, r in zip(self.potentials, range(len(self.potentials))):
            spectra_path = specfiles[r]
            
            df = self._single_time_spectra(spectra_path, smooth=smooth, digits=round_wl)
            self.spectra_vs_time[v] = df

        if any(droptimes):
            for st in self.spectra_vs_time:
                self.spectra_vs_time[st] = self.spectra_vs_time[st].drop(droptimes, axis=1)

        self.time_index()

        return

    def _single_time_spectra(self, spectra_path, smooth=3, digits=None):
        '''
        Generates the time-dependent spectra for a single dataframe.
        This is used internally to generate the dataFrame then passed to time_dep_spectra()
        
        spectra_path : str
            Path to a specific spectra file
            
        smooth : int, optional
            For smoothing the data via a boxcar filter. None = no smoothing. 
        
        Returns
        ---------
        df : dataFrame
            dataFrame of index = wavelength, columns = times, data = absorbance
            
        '''

        pp = pd.read_csv(spectra_path, sep='\t')

        try:
            runs = np.unique(pp['Spectrum number'])
        except:
            wl = pp['Wavelength (nm)'][0]
            runs = np.arange(1, len(np.where(pp['Wavelength (nm)'] == wl)[0]) + 1)

        times = np.unique(pp['Time (s)'])
        times = times - times[0]
        per_run = int(len(pp) / runs[-1])
        wl = pp['Wavelength (nm)'][0:per_run]

        # Set up dataframe
        df = pd.DataFrame(index=wl)
        if digits:
            df = df.set_index(np.round(df.index.values, digits)) # rounds wavelengths

        for k, t in zip(runs, times):

            try:
                data = pp[pp['Spectrum number'] == k]['Absorbance'].values
            except:
                idx = per_run * (k - 1)
                data = pp['Absorbance'].iloc[idx:idx + per_run].values

            if smooth:
                data = sg.fftconvolve(data, np.ones(smooth) / smooth, mode='same')

            df[np.round(t, 2)] = pd.Series(data, index=df.index)

        return df

    def spec_echem_voltage(self, time=0, smooth=3, digits=None):
        '''
        Takes the list of spectra files specfiles, then extracts the time-slice of 
        spectra from each file and returns as a single dataframe
        
        Also extracts the absorbance vs voltage at a particular wavelength
        
        time : int, optional
            What time slice to return

        smooth : int
            simple boxcar smooth of data for plotting/analysis purposes, controls
            size of filter 
         
        digits : int
            rounds the wavelengths to this number of decimal points
            
        Saves:
        ------
        spectra : time=0 spectra at each voltage
        spectra_sm : time=0 spectra (smoothed) at each voltage
            
        '''

        wl = self.spectra_vs_time[0].index.values

        # Set up dataFrame
        df = pd.DataFrame(index=wl)

        # Set up a "smoothed" dataFrame
        dfs = pd.DataFrame(index=wl)

        for v in self.spectra_vs_time:
            
            col = np.searchsorted(self.spectra_vs_time[v].columns.values, time)
            col = self.spectra_vs_time[v].columns.values[col]
            
            data = self.spectra_vs_time[v][col]
            df[v] = pd.Series(data, index=df.index, name=str(v))
            data = sg.fftconvolve(data, np.ones(smooth) / smooth, mode='same')
            dfs[v] = pd.Series(data, index=df.index, name=str(v))

        self.spectra = df
        self.spectra_sm = dfs

        return

    def time_index(self, stepfiles=None):
        '''
        Sets up the time index by reading from the first working electrode current file
        
        Alternatively, can read from the time-dependent spectra dataframe
        '''
        if stepfiles:
            pp = pd.read_csv(stepfiles, sep='\t')

            self.tx = pp['Corrected time (s)'].values
            self.tx = np.round(self.tx, 2)
        
        else:
                
            key = next(iter(self.spectra_vs_time)) # random key
            self.tx = self.spectra_vs_time[key].columns.values
        
        return
    

    def current_vs_time(self, stepfiles):
        '''
        Processes "step" files to generate the current vs time at each voltage

        Saves the integrated charge as well
        
        stepfiles : str, list
            List of steps files (containing working electrode current) on disk
        '''
        
        tx = []
        
        for fl, v in zip(stepfiles, self.potentials):
            
            pp = pd.read_csv(fl, sep='\t')
            data = pp['WE(1).Current (A)']
            
            if not any(tx):
                tx = pp['Corrected time (s)']
                df = pd.DataFrame(index=np.round(tx, 2))
            
            df[v] = pd.Series(data.values, index=df.index)

        self.current = df
      
        charge = pd.DataFrame(columns=self.current.columns, index = [0])
        charge.columns.name = 'Potential (V)'
        tx = self.current.index.values
        for p in self.current:
            charge[p] = spint.trapz(self.current.index.values, self.current[p].values)*1e3
        
        self.charge = charge

        return

    def single_wl_time(self, potential=0.9, wavelength=800, smooth=3):
        '''
        Extracts the time-dependent data from a single wavelength
        
        potential : float
            Find run corresponding to potential. Note in UV-Vis substrate is biased, not gate electrode
            
        wavelength : int, float
            Wavelength to extract. This will search for nearest wavelength row
            
        '''
        df = self.spectra_vs_time[potential].copy(deep=True)

        idx = df.index
        wl = idx.searchsorted(wavelength)

        data = df.loc[idx[wl]] - np.min(df.loc[idx[wl]])
        data = data / np.max(data)

        self.time_spectra = df.loc[idx[wl]]
        self.time_spectra_norm = pd.Series(data.values, index=df.loc[idx[wl]].index)

        # smooth
        for c in df.columns:
            df[c] = sg.fftconvolve(df[c], np.ones(smooth) / smooth, mode='same')

        data = df.loc[idx[wl]] - np.min(df.loc[idx[wl]])
        data = data / np.max(data)

        self.time_spectra_sm = df.loc[idx[wl]]
        self.time_spectra_norm_sm = pd.Series(data.values, index=df.loc[idx[wl]].index)

        return

    def abs_vs_voltage(self, wavelength=800, time=0):
        '''
        Extracts the absorbance vs voltage at a particular wavelength (threshold visualizing)
        '''
        tx = self.tx.searchsorted(time)
        if time == -1:
            tx = self.tx[-1]
        # self.vt = self.spectra.loc[idx[wl]]

        vt = []
        vt = pd.DataFrame(columns=['Abs'])
        for dv in self.spectra_vs_time:
            
            df = self.spectra_vs_time[dv]
            wl = df.index.values.searchsorted(wavelength)
            vt.loc[dv] = self.spectra_vs_time[dv].iloc[wl][tx]
            
        self.vt = vt

        return

    def volt(self, bias):
        '''
        returns voltage from potential list
        
        '''
        out = np.searchsorted(self.potentials, bias)

        return out

    def banded_fits(self, wl_start=700, wl_stop=900, voltage=1, fittype='exp'):
        '''
        Returns the fits from a range of spectra_vs_time data for a particular potential
        
        wl_start : int
        wl_stop : int
            The start and stop wavelengths for generating fits
            
        voltage : float
            The potential data to analyze in the spectra_vs_time dataFrame
            
        fittype: str
            of 'exp' 'biexp' and 'stretched', the form of the fitting function
            exp = single exponential (fastest)
            biexp = two exponentials
            stretched = stretched expontential
            
        Generates
        -------
        fits : ndarray list
            Contains the fit values. Either a single entry list for exp or a list of
                tuples for biexp and stretched
        
        '''

        wl_x = self.spectra_vs_time[voltage][wl_start:wl_stop]
        tx = self.time_spectra_norm_sm.index.values

        if fittype not in ['exp', 'biexp', 'stretched']:
            raise ValueError('Fit must be exp, biexp, or stretched')

        fits = []  # single exponential

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

        self.fits = np.array(fits)

        return


def fit_exp(t, y0, A, tau):
    return y0 + A * np.exp(-t / tau)


def fit_biexp(t, y0, A1, tau1, A2, tau2):
    return y0 + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)


def fit_strexp(t, y0, A, tau, beta):
    return y0 + A * (np.exp(-t / tau)) ** beta


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


def plot_spectra(uv, ax=None, smooth=False, crange=[0.2, 0.75], title=None, **kwargs):
    '''
    Simple plot of the spectra
    :param uv: uvvis Class object
    :param ax: Axes object
        if None, creates Figure
    :param crange: 2-size array
        Controls the color-range for generating the colormap
    :param title: str
        Image title
    :param smooth: bool
        Whether to use the smoothed or raw spectra

     The time slice printed is dependent on how the data are processed (default is t=0 s)
    '''

    cm = np.linspace(crange[0], crange[1], len(uv.spectra_sm.columns))

    if ax == None:
        fig, ax = plt.subplots(nrows=1, figsize=(12, 6), facecolor='white')

    if smooth:
        for i, cl in zip(uv.spectra_sm, cm):
            uv.spectra_sm[i].plot(ax=ax, linewidth=3, color=plt.cm.bone(cl), **kwargs)
    else:
        for i, cl in zip(uv.spectra_sm, cm):
            uv.spectra[i].plot(ax=ax, linewidth=3, color=plt.cm.bone(cl), **kwargs)

    ax.legend(labels=uv.potentials)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance (a.u.)')
    ax.set_title(title)

    return ax


def plot_spectra_vs_time(uv, ax=None, crange=[0.2, 0.75], potential=0.7, **kwargs):
    '''
    Plots the spectra vs time
    :param uv: uvvis Class object
    :param ax: Axes object
        if None, creates Figure
    :param crange: 2-size array
        Controls the color-range for generating the colormap
    :param kwargs:
    :return:
    '''
    endtime = uv.spectra_vs_time[potential].columns[-1]
    cm = np.linspace(crange[0], crange[1], len(uv.spectra_vs_time[potential].columns))

    if ax == None:
        fig, ax = plt.subplots(nrows=1, figsize=(12, 6), facecolor='white')

    for i, cl in zip(uv.spectra_vs_time[potential], cm):
        ax.plot(uv.spectra_vs_time[potential][i], color=plt.cm.bone(cl))

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance (a.u.)')
    ax.set_title(str(potential) + ' V kinetics over ' + str(endtime) + ' s')

    return ax


def plot_voltage(uv, ax=None, norm=None, wavelength=800, time=-1, 
                 flip_x = False, **kwargs):
    '''

    Plots the "threshold" from the absorbance.
    norm = normalize the threshold UV-Vis data

    wavelength : float
    t : float
        The time slice to plot, in seconds. -1 is the final time

    '''
    if ax == None:
        fig, ax = plt.subplots(nrows=1, figsize=(12, 6), facecolor='white')

    if flip_x:
        flip = -1
    else:
        flip = 1

    uv.abs_vs_voltage(wavelength=wavelength, time=time)

    if norm == None:
        ax.plot(uv.vt.index.values * flip, **kwargs)
    else:
        numerator = (uv.vt.values - uv.vt.values.min())
        ax.plot(uv.vt.index.values * flip, numerator / numerator.max(), **kwargs)
    ax.set_xlabel('Gate Bias (V)')
    ax.set_ylabel('Absorbance (a.u.)')
    ax.set_title('Absorbance vs Voltage at ' + str(wavelength) + ' nm')

    return ax


def spectrogram(uv, potential=0.8, **kwargs):
    fig, ax = plt.subplots(nrows=2, figsize=(12, 18))

    if 'cmap' not in kwargs:
        # kwargs['cmap'] = 'BrBG_r'
        kwargs['cmap'] = 'icefire'

    wl = np.round(uv.spectra_vs_time[potential].index.values, 2)
    df = pd.DataFrame.copy(uv.spectra_vs_time[potential])
    df = df.set_index(wl)

    sns.heatmap(df, ax=ax[0], **kwargs)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Wavelength (nm)')

    df = pd.DataFrame.copy(uv.spectra_sm)
    df = df.set_index(wl)
    sns.heatmap(df, ax=ax[1], **kwargs)
    ax[1].set_xlabel('Voltage (V)')
    ax[1].set_ylabel('Wavelength (nm)')

    return ax

def save_h5(data, filename):
    '''
    Saves the to two HDF5 files (.h5)
    '''
    
    if isinstance(filename, str):
        filename = Path(filename)
    
    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('potentials', (len(data.potentials),))
        dset[:] = data.potentials[:]
        try:
            dset = f.create_dataset('charge', (len(data.charge.values[0]),))
            dset[:] = data.charge.values[0][:]
        except:
            pass
    f.close()
    
    for p in data.spectra_vs_time:
        data.spectra_vs_time[p].to_hdf(filename, key=str(p), mode='a')
    
    try:
        data.current.to_hdf(filename, key ='current', mode='a')
    except:
        pass
    
    return

def convert_h5(h5file):
    '''
    Converts a saved hdf5 to uvvis Class format
    
    axis0 = time
    axis1 = wavelength
    block0_items
    '''
    data = uv_vis(None,None,None)
    file =  h5py.File(h5file, 'r')
    data.potentials = file['potentials'][()]
    
    folders = []
    for f in file:
        folders.append(f)
    non_potentials = ['current', 'charge', 'potentials']
    _fold_temp = [c if c not in non_potentials else None for c in folders]
    folders = [c for c in _fold_temp if(c)]            
    
    try:
        folders_num = [float(p) for p in folders[:]]
    except: # for old 'x-1.0V' style, crops 'x'
        folders_num = [float(p[1:]) for p in folders[:]]
    
    # The spectra_vs_time data
    df_dict = {}
    for v, n in zip(folders, folders_num):
        try:
            spec_file = file[v]
        except:
            p = 'x'+v
            spec_file = file[p]
        
        df = pd.DataFrame(data = spec_file['block0_values'][()], 
                          index = spec_file['axis1'][()], 
                          columns = spec_file['axis0'])
        df.index.name = 'Wavelength (nm)'
        df.columns.name = 'Time (s)'
        df_dict[n] = df
        data.tx = np.round(spec_file['axis0'], 2)
        
    data.spectra_vs_time = df_dict
    
    # Now get the current data
    current = pd.DataFrame(data = file['current']['block0_values'][()],
                           index = file['current']['axis1'][()],
                           columns = file['current']['axis0'])
    current.index.name = 'Time (s)'
    current.columns.name = 'Potential (V)'
        
    data.current = current
    
    data.charge = pd.DataFrame(data = file['charge'][()],
                               index = data.potentials.T)
    
    file.close()
    
    return data
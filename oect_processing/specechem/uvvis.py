# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:36:20 2018

@author: Raj
"""

import numpy as np
import pandas as pd
from scipy import integrate as spint
from scipy import signal as sg
from scipy.optimize import curve_fit

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


class UVVis(object):

    def __init__(self, steps=None, specs=None, potentials=None):
        '''
        Parameters
        ----------
        steps : list of str
            Paths to step (working electrode current) files.
        specs : list of str
            Paths to spectra files.
        potentials : list of float
            Applied voltages corresponding to each step/spectra file.

        Attributes
        ----------
        spectra : DataFrame
            Time-zero spectra at each voltage.
        spectra_sm : DataFrame
            Smoothed time-zero spectra at each voltage.
        spectra_vs_time : dict
            Time-dependent spectra at each voltage; e.g. spectra_vs_time[0.9]
            is a DataFrame with wavelength as index and time as columns.
        current : DataFrame
            Time-resolved current at each voltage step.
        time_spectra : Series
            Absorbance vs time at a chosen wavelength and potential.
        time_spectra_norm : Series
            Normalised version of time_spectra.
        vt : Series
            Absorbance vs voltage at a particular wavelength (threshold visualisation).
        tx : ndarray
            Time axis used for plotting and fitting.
        fits : ndarray
            Exponential fit results from banded_fits.
        '''
        self.steps = steps
        self.specs = specs
        self.potentials = potentials

        return

    def time_dep_spectra(self, specfiles, smooth=None, round_wl=2, droptimes=None):
        '''
        Builds spectra_vs_time: a dict of time-dependent spectra DataFrames per voltage.

        e.g. spectra_vs_time[0.9] has wavelength as index and time slices as columns.
        This dict is the primary data structure of the class.

        Parameters
        ----------
        specfiles : list of str
            Paths to spectra files on disk.
        smooth : int, optional
            Boxcar filter width for smoothing. None = no smoothing. Typical value: 3.
        round_wl : int, optional
            Number of decimal places to round wavelength values. None = no rounding.
        droptimes : list or array, optional
            Time indices to drop (e.g. to remove artefacts at start or end of a run).
        '''

        self.spectra_vs_time = {}
        for v, r in zip(self.potentials, range(len(self.potentials))):
            spectra_path = specfiles[r]

            df = self._single_time_spectra(spectra_path, smooth=smooth, digits=round_wl)

            self.spectra_vs_time[v] = df

        if droptimes:
            for st in self.spectra_vs_time:
                self.spectra_vs_time[st] = self.spectra_vs_time[st].drop(droptimes, axis=1)

        self.time_index()

        return

    def _single_time_spectra(self, spectra_path, smooth=3, digits=None):
        '''
        Builds a single time-dependent spectra DataFrame from one file.

        Parameters
        ----------
        spectra_path : str
            Path to a spectra file.
        smooth : int, optional
            Boxcar filter width. None = no smoothing.
        digits : int, optional
            Rounds wavelength index to this many decimal places.

        Returns
        -------
        DataFrame
            Index = wavelength (nm), columns = time (s), values = absorbance.
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

        index = np.round(wl.values, digits) if digits else wl.values

        cols = {}
        for k, t in zip(runs, times):

            try:
                data = pp[pp['Spectrum number'] == k]['Absorbance'].values
            except:
                idx = per_run * (k - 1)
                data = pp['Absorbance'].iloc[idx:idx + per_run].values

            if smooth:
                data = sg.fftconvolve(data, np.ones(smooth) / smooth, mode='same')

            cols[np.round(t, 2)] = data

        df = pd.DataFrame(cols, index=index)

        return df

    def spec_echem_voltage(self, time=0, smooth=3, digits=None):
        '''
        Extracts a single time slice from each voltage's spectra and builds spectra/spectra_sm.

        Parameters
        ----------
        time : int, optional
            Time slice (s) to extract from each voltage's spectra.
        smooth : int, optional
            Boxcar filter width for spectra_sm.
        digits : int, optional
            Rounds wavelength index to this many decimal places.

        Notes
        -----
        Saves results to self.spectra (unsmoothed) and self.spectra_sm (smoothed).
        '''

        wl = self.spectra_vs_time[self.potentials[0]].index.values

        cols = {}
        cols_sm = {}

        for v in self.spectra_vs_time:
            col = np.searchsorted(self.spectra_vs_time[v].columns.values, time)
            col = self.spectra_vs_time[v].columns.values[col]

            data = self.spectra_vs_time[v][col]
            cols[v] = data.values
            cols_sm[v] = sg.fftconvolve(data, np.ones(smooth) / smooth, mode='same')

        self.spectra = pd.DataFrame(cols, index=wl)
        self.spectra_sm = pd.DataFrame(cols_sm, index=wl)

        return

    def time_index(self, stepfiles=None):
        '''
        Sets self.tx (the time axis) from a step file or from spectra_vs_time.

        Parameters
        ----------
        stepfiles : str, optional
            Path to a step file. If None, reads time axis from spectra_vs_time.
        '''
        if stepfiles:
            pp = pd.read_csv(stepfiles, sep='\t')

            self.tx = pp['Corrected time (s)'].values
            self.tx = np.round(self.tx, 2)

        else:

            key = next(iter(self.spectra_vs_time))  # random key
            self.tx = self.spectra_vs_time[key].columns.values

        return

    def current_vs_time(self, stepfiles):
        '''
        Loads step files to build self.current (current vs time) and self.charge (integrated).

        Parameters
        ----------
        stepfiles : list of str
            Paths to working electrode current files, one per voltage step.
        '''

        tx = []
        cols = {}

        for fl, v in zip(stepfiles, self.potentials):

            pp = pd.read_csv(fl, sep='\t')

            if not any(tx):
                tx = np.round(pp['Corrected time (s)'].values, 2)

            cols[v] = pp['WE(1).Current (A)'].values

        self.current = pd.DataFrame(cols, index=tx)

        charge = pd.DataFrame(columns=self.current.columns, index=[0])
        charge.columns.name = 'Potential (V)'
        tx = self.current.index.values
        for p in self.current:
            charge[p] = spint.trapezoid(self.current[p].values, x=self.current.index.values) * 1e3

        self.charge = charge

        return

    def single_wl_time(self, potential=0.9, wavelength=800, smooth=3):
        '''
        Extracts normalised absorbance vs time at a single wavelength and potential.

        Parameters
        ----------
        potential : float, optional
            Voltage key in spectra_vs_time. Note: substrate is biased in UV-Vis, not gate.
        wavelength : int or float, optional
            Wavelength (nm) to extract; nearest available row is used.
        smooth : int, optional
            Boxcar filter width for smoothed output.
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
        Extracts absorbance vs voltage at a fixed wavelength and time slice.

        Parameters
        ----------
        wavelength : int or float, optional
            Wavelength (nm) to extract; nearest available row is used.
        time : float, optional
            Time slice (s) to use. -1 uses the final time point.
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
        Returns the index of a voltage in self.potentials.

        Parameters
        ----------
        bias : float
            Voltage to search for.

        Returns
        -------
        int
            Index of the nearest voltage in self.potentials.
        '''
        out = np.searchsorted(self.potentials, bias)

        return out

    def banded_fits(self, wl_start=700, wl_stop=900, voltage=1, fittype='exp'):
        '''
        Fits time-resolved absorbance at each wavelength in a band for a given potential.

        Parameters
        ----------
        wl_start : int, optional
            Start wavelength (nm) of the fitting band.
        wl_stop : int, optional
            Stop wavelength (nm) of the fitting band.
        voltage : float, optional
            Potential key in spectra_vs_time to analyse.
        fittype : str, optional
            Fitting function: 'exp' (single exponential, fastest),
            'biexp' (two exponentials), or 'stretched' (stretched exponential).

        Notes
        -----
        Saves results to self.fits as an ndarray. Single values for 'exp',
        tuples for 'biexp' and 'stretched'.
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
    '''
    Single exponential decay: y0 + A * exp(-t / tau).

    Parameters
    ----------
    t : array-like
        Time values.
    y0 : float
        Baseline offset.
    A : float
        Amplitude.
    tau : float
        Time constant.

    Returns
    -------
    ndarray
    '''
    return y0 + A * np.exp(-t / tau)


def fit_biexp(t, y0, A1, tau1, A2, tau2):
    '''
    Bi-exponential decay: y0 + A1*exp(-t/tau1) + A2*exp(-t/tau2).

    Parameters
    ----------
    t : array-like
        Time values.
    y0 : float
        Baseline offset.
    A1 : float
        Amplitude of first component.
    tau1 : float
        Time constant of first component.
    A2 : float
        Amplitude of second component.
    tau2 : float
        Time constant of second component.

    Returns
    -------
    ndarray
    '''
    return y0 + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)


def fit_strexp(t, y0, A, tau, beta):
    '''
    Stretched exponential decay: y0 + A * exp(-t/tau)^beta.

    Parameters
    ----------
    t : array-like
        Time values.
    y0 : float
        Baseline offset.
    A : float
        Amplitude.
    tau : float
        Time constant.
    beta : float
        Stretching exponent.

    Returns
    -------
    ndarray
    '''
    return y0 + A * (np.exp(-t / tau)) ** beta

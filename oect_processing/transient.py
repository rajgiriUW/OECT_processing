# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:15:37 2019

@author: GingerLab
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def read_time_dep(path, start=0, stop=0, v_limit=None, skipfooter=1):
    '''
    Reads in the time-dependent data using Raj's automated version
    Saves all the different current 
    
    :param path:
    :type path: String
    
    :param start: Time index (in ms) to set as t=0 for the data
    :type start: int
    
    :param stop: Time index (in ms) to set as the end point for the data
    :type stop: int
        
    :param v_limit: Limits constant current data to when the voltage_compliance limit is reached
        Do not use this for Constant Voltage data
        e.g. -0.9 V is typical
    :type v_limit: float
    
    :param skipfooter:
    :type skipfooter: int, optional
        
    :returns: Single DataFrame with all the indices corrected
    :rtype: DataFrame
    
    '''
    df = pd.read_csv(path, sep='\t', skipfooter=skipfooter, engine='python')
    df = df.set_index('Time (ms)')  # convert to seconds

    # crop the pre-trigger stuff
    f = df.index.searchsorted(start)
    df = df.drop(df.index.values[:f])
    if stop != 0:
        f = df.index.searchsorted(stop)
        df = df.drop(df.index.values[f:])
    if v_limit:
        df = df.loc[np.abs(df['V_G(V)']) <= np.abs(v_limit)]

    # Add the setpoints as attributes for easy reference
    if df.columns[0] == 'I_G (A)':
        df.is_cc = True
        df.is_cv = False
    elif df.columns[0] == 'V_G (V)':
        df.is_cc = False
        df.is_cv = True

    df.name = path.split(r'/')[-1][:-4]
    ids = df['I_DS (A)']
    norm = 1 + (ids - np.min(np.abs(ids))) / (np.max(np.abs(ids)) - np.min(np.abs(ids)))
    df['I_DS_norm (a.u.)'] = norm

    return df


def plot_current(df, ax=None, v_comp=-1, norm=False, plot_voltage=True):
    '''
    Plot the current data
    Only plots the current where voltage doesn't saturate  (v_comp)
 
    :param df:
    :type df: DataFrame
    
    :param ax:
    :type ax: matplotlib Axes object, optional   
 
    :param v_comp: The voltage compliance limit during constant_current traces
        This has no real effect in a usual constant_voltage case
    :type v_comp: float
    
    :param norm: Normalize the current
    :type norm: bool, optional
    
    :param plot_voltage: Plot the voltage (only useful for constant voltage plotting)
    :type plot_voltage: bool, optional
    
    :returns: tuple (fig, ax)
        WHERE
        [type] fig is...
        [type] ax is...
    '''
    if not ax:
        fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')

    if df.is_cc:
        yy = df.loc[np.abs(df['V_G (V)']) <= np.abs(v_comp)]
        xx = df.loc[np.abs(df['V_G (V)']) <= np.abs(v_comp)].index.values
    else:
        yy = df
        xx = df.index.values

    if norm:
        ax.plot(xx / 1000, yy['I_DS_norm (a.u.)'], 'b')
        ax.set_ylabel('Norm. I$_{ds}$ (a.u.)')
    else:
        ax.plot(xx / 1000, yy['I_DS (A)'], 'b')
        ax.set_ylabel('I$_{ds}$ (mA)')

    if plot_voltage:
        ax2 = ax.twinx()
        ax2.plot(xx / 1000, yy['V_G (V)'], 'r--')
        ax2.set_ylabel('Voltage (V)', rotation=270, labelpad=10)

    ax.set_xlabel('Time (s)')
    ax.set_title(df.name)
    plt.tight_layout()

    return fig, ax


def expf(t, y0, A, tau):
    '''
    :param t:
    :type t: float
    
    :param y0:
    :type y0: float
    
    :param A:
    :type A: float
    
    :param tau:
    :type tau: float
    
    :returns:
    :rtype: float
    '''
    return y0 + A * np.exp(-t / tau)


def fit_cycles(df, doping_time, dedoping_time, func=expf, norm=False,
               plot=True, p_type=True, dope_tau_p0=10, dedope_tau_p0=1):
    '''
    Simple single exponential fitting of kinetics
    
    :param df: The DataFrame containing our kinetics information
    :type df: Pandas DataFrame
        
    :param doping_time: The time (seconds) the sample is doped
    :type doping_time: int
        
    :param dedoping_time: The time (seconds) the sample is dedoped
    :type dedoping_time: int
        
    :param func: default = single exponential. The fitting function to use. 
        Currently this is fixed for dopnig and dedoping
    :type func: function

    :param norm: If True, normalizes the current data prior to fitting
        Sometimes needed for very small currents
    :type norm: bool, optional
        
    :param plot: Displays the current and then overlays the fit per cycle
    :type plot: bool, optional
        
    :param p_type: Assumes doping is a negative voltage
    :type p_type: bool, optional
        
    :param dope_tau_p0: The initial guess for the doping tau
    :type dope_tau_p0: float, optional
        
    :param dedope_tau_p0: The initial guess for the dedoping tau
    :type dedope_tau_p0: float, optional
    
    :returns: tuple (doping_fits, dedoping_fits)
        WHERE
        [type] doping_fits is...
        [type] dedoping_fits is...
        
    '''

    t = df.index.values / 1000  # in seconds

    # cycle_idx contains indices for each cycle time in the DataFrame
    # doping_idx contains the indices where doping stops in each cycle
    if p_type:
        doping_idx = np.where(np.diff(df['V_G (V)']) < 0)[0]
        dedoping_idx = np.where(np.diff(df['V_G (V)']) > 0)[0]
    else:
        doping_idx = np.where(np.diff(df['V_G (V)']) > 0)[0]
        dedoping_idx = np.where(np.diff(df['V_G (V)']) < 0)[0]

    doping_fits = []
    dedoping_fits = []

    if plot:
        _, ax = plot_current(df, norm=norm, plot_voltage=(False))

    for n, _ in enumerate(doping_idx):

        _dope = range(doping_idx[n], dedoping_idx[n] + 1)

        try:
            _dedope = range(dedoping_idx[n], doping_idx[n + 1] + 1)
        except:
            _dedope = range(dedoping_idx[n], len(t))
        # X-axis times
        xx_dope = t[_dope]
        xx_dedope = t[_dedope]

        if norm:
            yy_dope = df['I_DS_norm (a.u.)'].iloc[_dope]
            yy_dedope = df['I_DS_norm (a.u.)'].iloc[_dedope]
        else:
            yy_dope = df['I_DS (A)'].iloc[_dope]
            yy_dedope = df['I_DS (A)'].iloc[_dedope]

        # Curve fitting
        try:
            popt_dope, _ = curve_fit(func, xx_dope - xx_dope[0], yy_dope,
                                     p0=[yy_dope.iloc[0],
                                         yy_dope.max() - yy_dope.min(),
                                         dope_tau_p0])
            popt_dedope, _ = curve_fit(func, xx_dedope - xx_dedope[0], yy_dedope,
                                       p0=[yy_dedope.iloc[0],
                                           -(yy_dedope.max() - yy_dedope.min()),
                                           dedope_tau_p0])
        except:
            print('Error when curve fitting cycle #', n)
            popt_dope = [0, 0, 0]
            popt_dedope = [0, 0, 0]

        doping_fits.append(popt_dope)
        dedoping_fits.append(popt_dedope)

        if plot:
            ax.plot(xx_dope, func(xx_dope - xx_dope[0], *popt_dope), 'r--')
            ax.plot(xx_dedope, func(xx_dedope - xx_dedope[0], *popt_dedope), 'g--')

    return doping_fits, dedoping_fits

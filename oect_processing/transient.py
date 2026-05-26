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
    Reads time-dependent OECT transient data from a tab-delimited file.

    Parameters
    ----------
    path : str
        Path to the data file.
    start : int, optional
        Time index (ms) to set as t=0.
    stop : int, optional
        Time index (ms) to use as the endpoint. 0 means use all data.
    v_limit : float, optional
        Clips constant-current data at voltage compliance (e.g. -0.9 V).
        Do not use for constant-voltage data.
    skipfooter : int, optional
        Number of footer rows to skip when reading.

    Returns
    -------
    DataFrame
        Single DataFrame with corrected time index and normalised current column.
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
    Plots drain current vs time, optionally with gate voltage on a right axis.

    Parameters
    ----------
    df : DataFrame
        Transient data from read_time_dep.
    ax : matplotlib Axes, optional
        Existing axes to plot on. Creates a new figure if None.
    v_comp : float, optional
        Voltage compliance limit for constant-current traces. Data beyond this
        voltage are excluded. Has no effect for constant-voltage data.
    norm : bool, optional
        If True, plots normalised current instead of raw current.
    plot_voltage : bool, optional
        If True, overlays gate voltage on a right axis.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
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
    Single exponential decay function.

    Parameters
    ----------
    t : float or array-like
        Time values.
    y0 : float
        Offset (baseline).
    A : float
        Amplitude.
    tau : float
        Time constant.

    Returns
    -------
    float or ndarray
    '''
    return y0 + A * np.exp(-t / tau)


def fit_cycles(df, doping_time, dedoping_time, func=expf, norm=False,
               plot=True, p_type=True, dope_tau_p0=10, dedope_tau_p0=1):
    '''
    Fits doping and dedoping kinetics cycle-by-cycle with a single exponential.

    Parameters
    ----------
    df : DataFrame
        Transient kinetics data from read_time_dep.
    doping_time : int
        Duration (seconds) of each doping pulse.
    dedoping_time : int
        Duration (seconds) of each dedoping pulse.
    func : callable, optional
        Fitting function; defaults to single exponential expf.
    norm : bool, optional
        If True, normalises current prior to fitting (useful for small currents).
    plot : bool, optional
        If True, overlays fits on the current trace.
    p_type : bool, optional
        If True, assumes doping corresponds to a negative gate voltage.
    dope_tau_p0 : float, optional
        Initial guess for the doping time constant (seconds).
    dedope_tau_p0 : float, optional
        Initial guess for the dedoping time constant (seconds).

    Returns
    -------
    doping_fits : list
        Fit parameters for each doping half-cycle.
    dedoping_fits : list
        Fit parameters for each dedoping half-cycle.
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

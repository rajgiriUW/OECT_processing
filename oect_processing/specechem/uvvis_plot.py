import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_time(uv, ax=None, norm=True, smooth=False, **kwargs):
    '''
    :param uv:
    :type uv: uvvis class object
    
    :param ax:
    :type ax:
    
    :param norm:
    :type norm: bool
    
    :param smooth:
    :type smooth: bool
    
    :param kwargs: matplotlib kwargs
    :type kwargs: dict
    
    :returns:
    :rtype:
    '''
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
    The time slice printed is dependent on how the data are processed (default is t=0 s)
    
    :param uv:
    :type uv: UVVis Class object
    
    :param ax: if None, creates Figure
    :type ax: Axes object
    
    :param smooth: Whether to use the smoothed or raw spectra
    :type smooth: bool
    
    :param crange: Controls the color-range for generating the colormap
    :type crange: 2-size array
     
    :param title: Image title
    :type title: str
    
    :param kwargs: matplotlib kwargs
    :type kwargs: dict
    
    :returns:
    :rtype:
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
    :param uv:
    :type uv: UVVis Class object
    
    :param ax: if None, creates Figure
    :type ax: Axes object
    
    :param crange:  Controls the color-range for generating the colormap
    :type crange: 2-size array
    
    :param potential: Valid column in the UVVis object, chooses which potential to plot
    :type potential: float
        
    :param kwargs: matplotlib kwargs
    :type kwargs: dict
    
    :returns:
    :rtype:
        
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
                 flip_x=False, **kwargs):
    '''
    Plots the "threshold" from the absorbance.

    :param uv:
    :type uv: UVVis Class object
    
    :param norm: normalize the threshold UV-Vis data
    :type norm:

    :param wavelength:
    :type wavelength: float
    
    :param time: The time slice to plot, in seconds. -1 is the final time
    :type time: float
    
    :param flip_x:
    :type flip_x: bool
    
    :param kwargs: matplotlib kwargs
    :type kwargs: dict
    
    :returns:
    :rtype:
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
    '''
    :param uv:
    :type uv: UVVis Class object
    
    :param potential: Valid column in the UVVis object, chooses which potential to plot
    :type potential: float
    
    :param kwargs: matplotlib kwargs
    :type kwargs: dict
    '''
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

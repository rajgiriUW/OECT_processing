# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:58:00 2019

@author: Raj
"""

import pandas as pd
from matplotlib import pyplot as plt


def read_eis(path):
    '''
    Note that you probably need to recast this as a float
    
    Some notes on the data:
        As saved, these are saving BOTH the measurement and the simulation fit
        Columns 3-9 are the measured data
        Columns 10+ are the fit data (appended with a .1, e.g. Phase.1)
        
    Returns:
        edv : Dict
            Dict with each voltage as a key (e.g. edv[0.7] is the 0.7 V run)
    '''

    edv = {}
    df = pd.read_csv(path, sep='\t')

    # finds the string locations and assigns the segments to the dicitonary
    # The +1 and +2 are due to the way extraneous values are saved in 
    nm = []
    volts = []
    c = df['Column 1'].values
    for k, x in zip(c, range(len(c))):
        if isinstance(k, str):
            if 'Column' not in k:
                nm.append(x + 1)
                volts.append(float(df.iloc[x]['Column 2 (V)']))
    nm.append(len(c) + 2)
    for v, x in zip(volts, range(len(volts))):
        edv[v] = df.iloc[nm[x] - 1:nm[x + 1] - 2]
        edv[v] = edv[v].drop(columns=['Column 1', 'Column 2 (V)'])
        edv[v] = edv[v].astype(float)

    return edv


def plot_bode(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df['Frequency (Hz)'], df['Z (Ω)'], 'bo')
    ax.set_xscale('log')
    ax2 = ax.twinx()
    ax2.plot(df['Frequency (Hz)'], df['-Phase (°)'], 'r^')
    ax2.set_xscale('log')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Z (Ω)')
    ax2.set_ylabel('-Phase (°)')

    return


def plot_nyquist(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df["Z' (Ω)"], df["-Z'' (Ω)"], 'bo')

    ax.set_xlabel("Z' (Ω)")
    ax.set_ylabel("-Z'' (Ω)")
    return

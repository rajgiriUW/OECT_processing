# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:15:37 2019

@author: GingerLab
"""

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

def read_time_dep(path):
    
    df = pd.read_csv(path, sep='\t')
    df = df.set_index('Time (s)')
    try:
        currents = pd.unique(df['Ig (A) '])
    except:
        currents = pd.unique(df['Ig (A)'])
        
    df.currents = currents
    
    return df

def find_turnon(df):
    
    npts = len(df.loc[df['Ig (A) '] == -1e-7])

    tx = df.index.values[:npts]
    
    # gradient
    diffy = np.gradient(df.iloc[:npts]['Ids (A)'])
    diffx = np.gradient(tx[:npts])
    diffy = diffy / diffx
    
    mx = np.argmax(diffy)
    
    return mx, npts

def max_dIds(df, mx, npts):
    
    return
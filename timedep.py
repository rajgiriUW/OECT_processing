# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:15:37 2019

@author: GingerLab
"""

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import pyplot as plt

def read_time_dep(path, croptime=30000):
    '''
    Reads in the time-dependent data using Raj's automated version
    Saves all the different current 
    
    croptime : int
        Time index (ms) to set as t=0 for the data
    
    '''
    df = pd.read_csv(path, sep='\t')
    df = df.set_index('Time (s)')
    try:
        currents = pd.unique(df['Setpoint'])
    except:
        currents = pd.unique(df['Ig (A) '])
        df = df.rename(columns={'Ig (A) ': 'Current (A)'})
    
    # crop the pre-trigger stuff
    for i in currents:
        f = df.loc[df['Setpoint'] == i].index.searchsorted(croptime)
        df = df.drop(df.loc[df['Setpoint'] == i].index[:f])

    # create a dict to separate the currents
    device = {}
    
    for i in currents:
        d = df.loc[df['Setpoint'] == i]
        device[i] = d

    df.currents = currents
    
    return df, device

def plot_current(df, norm=False):
    
    fig, ax = plt.subplots(figsize=(5,5))
    
    for i in df.currents:
        if norm:
            xx = df.loc[df['Setpoint'] == i]['Ids (A)'].index.values
            yy = df.loc[df['Setpoint'] == i]['Ids (A)'].values
            yy = (yy-np.min(yy))/(np.max(yy) - np.min(yy))
            ax.plot(xx, yy)
        else:
            ax.plot(df.loc[df['Setpoint'] == i]['Ids (A)'])

    ax.legend(labels = df.currents)

    return ax

def find_turnon(df, current= -1e-7):
    
    npts = len(df.loc[df['Setpoint'] == current])

    tx = df.index.values[:npts]
    
    # gradient
    diffy = np.gradient(df.iloc[:npts]['Ids (A)'])
    diffx = np.gradient(tx[:npts])
    diffy = diffy / diffx
    
    mx = np.argmax(diffy)
    
    return mx, npts

def crop_prepulse(df):
    '''
    df_total, device = timedep.crop_prepulse(df)
    
    df = dataframe from read_time_dep
    
    Crops all the data before the initial turn-on event. Manually shifts to 
    nearest 10000 ms point (assumes I set at an even second mark)
    
    df_total = big dataframe with all the data (doesn't standardize times)
    device = dictionary of currents
    '''
    
    df_total = pd.DataFrame()
    device = {}
    
    for i in df.currents:
        d = pd.DataFrame()
        mx, npts = find_turnon(df, i)
        print(i)
        f = int(np.floor(df.loc[df['Setpoint'] == i].index.values[mx]/10000))*10000
        if f == 0:
            f = 10000
        xx = df.loc[df['Setpoint'] == i].loc[f:].index.values
        yy = df.loc[df['Setpoint'] == i]['Ids (A)'].loc[f:].values
        d[i] = yy
        d = d.set_index(xx-xx[0])
        device[i] = d
    
    df_total = pd.concat([device[a] for a in device])
    df_total.currents = df.currents
    
    return df_total, device

def crop_fixed(df, timeon=10000):
    '''
    df_total, device = timedep.crop_prepulse(df)
    
    df = dataframe from read_time_dep
    
    Crops all the data before the initial turn-on event. Manually shifts to 
    nearest 10000 ms point (assumes I set at an even second mark)
    
    df_total = big dataframe with all the data (doesn't standardize times)
    device = dictionary of currents
    '''
    
    df_total = pd.DataFrame()
    device = {}
    
    for i in df.currents:
        d = pd.DataFrame()
        print(i)
        f = df.loc[df['Setpoint'] == i].index.searchsorted(timeon)

        xx = df.loc[df['Setpoint'] == i].iloc[f:].index.values
        yy = df.loc[df['Setpoint'] == i]['Ids (A)'].iloc[f:].values
        d[i] = yy
        d = d.set_index(xx-xx[0])
        device[i] = d
    
    df_total = pd.concat([device[a] for a in device])
    df_total.currents = df.currents
    
    return df_total, device

def line_f(x, a, b):

    return a + b * x


def bernards(t, del_I, f, tau_e, tau_i, y_err):
    '''
    Bernards model fitting
    Adv. Funct. Mater. 17, pp. 3538–3544 (2007)
    
    Note that this is the constant gate voltage model
    The constant gate current model is basically a line, which is only valid
     for very low gate currents (because voltage compliance)
    
    '''
    return y_err + del_I * (1 - f * tau_e/tau_i) * np.exp(-t/tau_i)

def friedlein(t, mu, Cg, L, Vg, Rg, Vt, Vd):
    '''
    Friedlein transient model
    Adv. Mater. 28, pp. 8398–8404 (2016)
    
    Assumes constant voltage step, not current
    '''
    
    return (mu*Cg/L**2) * (Vt - Vg * -np.expm1(-t/(Rg*Cg) ) - Vd/2) * Vd

def friedlein_sat(t, mu, Cg, L, Vg, Rg, Vt, Ierr):
    '''
    Friedlein transient model
    Adv. Mater. 28, pp. 8398–8404 (2016)
    
    Assumes constant voltage step, not current
    '''
    
    return (mu*Cg/L**2) * (Vg * -np.expm1(-t/(Rg*Cg)) - Vt)**2 + Ierr

def faria(t, I0, V0, gm, Rd, Rs, Cd, f):
    '''
    Faria Org Elec model
    Organic Electronics 45, pp. 215-221 (2015)
    '''
    
    Ig = V0*(gm * Rd - f)/(Rd + Rs)
    Ich = V0*Rd *(gm*Rs + f)/(Rs * (Rd + Rs)) * np.exp(-t*(Rd+Rs)/(Cd*Rd*Rs))
    
    return I0 + Ig - Ich

def fit_time(df, func='bernards', plot=True):
    
    xx = df.index.values / 1000.0
    
    yy = df.values[:,0]
    
    # Bernards model parameters
    y_err = 0
    del_I = -np.min(yy) # change in drain current 
    tau_e = 1e-5 # electronic response time
    tau_i = 1e-1 # ionic diffusion time
    
    # Friedlein model parameters
    mu = 1e-2
    L = 20e-6
    Cg = 1 # "ionic" capacitance, around 100 nF
    Vt = -0.4
    Vd = -0.6
    Rg = 1e3 # ionic resistance, 
    
    # Faria model
    I0 = yy[0] #initial current
    V0 = -0.85 #gate voltage
    gm = 1e-3 # 1 mS
    Rd = 100e3 # 1 kOhm, channel resistance
    Cd = 100e-3 # channel capacitance
    Rs = 2e3 # solution resistance
    f = 0.7
    
    if func is 'bernards':
        popt, _ = curve_fit(bernards, xx, yy, p0=[del_I, 0.5, tau_e, tau_i, y_err])
    elif func is 'friedlein':
        popt, _ = curve_fit(friedlein, xx, yy, p0=[mu, Cg, L, -0.8, Rg, Vt, Vd])
    elif func is 'faria':
        popt, _ = curve_fit(faria, xx, yy, p0=[I0, V0, gm, Rd, Rs, Cd, f])
    
    if plot:
        plt.figure()
        plt.plot(xx, yy, 'b-', linewidth=3)
        
        if func is 'bernards':
            plt.plot(xx, bernards(xx, *popt), 'r--', linewidth=3)
        elif func is 'friedlein':
            plt.plot(xx, friedlein(xx, *popt), 'r--', linewidth=3)
        elif func is 'faria':
            plt.plot(xx, faria(xx, *popt), 'r--', linewidth=3)
        
    
    return popt
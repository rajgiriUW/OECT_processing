# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:15:37 2019

@author: GingerLab
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import pandas as pd
from matplotlib import pyplot as plt

import lmfit


def read_time_dep(path, croptime=30, v_limit=None):
    '''
    Reads in the time-dependent data using Raj's automated version
    Saves all the different current 
    
    croptime : int
        Time index (s) to set as t=0 for the data
    v_limit : float
        Limits constant current data to when the voltage_compliance limit is reached
        Do not use this for Constant Voltage data
        e.g. -0.9 V is typical
        
    Returns:
        
    df : DataFrame
        Single DataFrame with all the indices corrected
    device : Dict
        Dict of the individual current DataFrames
    
    Both have equivalent data, it just can be easier to deal with dicts sometimes
    but a single DataFrame is convenient for SeaBorn plotting
    '''
    df = pd.read_csv(path, sep='\t')
    df = df.set_index('Time (s)')  # convert to seconds
    df = df.set_index(df.index.values / 1000.0)

    try:
        currents = pd.unique(df['Setpoint'])
    except:
        currents = pd.unique(df['Ig (A) '])
        df = df.rename(columns={'Ig (A) ': 'Current (A)'})

    # crop the pre-trigger stuff
    for i in currents:
        d = df.loc[df['Setpoint'] == i]
        f = d.index.searchsorted(croptime)
        df = df.drop(d.index[:f])
        if v_limit:
            df = df.loc[np.abs(df['Voltage (V)']) <= np.abs(v_limit)]

    # create a dict to separate the currents
    device = {}
    for i in currents:
        d = df.loc[df['Setpoint'] == i]
        d = d.set_index(d.index.values - d.index.values[0])
        device[i] = d

    # concatenate the corrected index dataframes into a single DataFrame
    _dv = [device[dv] for dv in device]
    df = pd.DataFrame()
    df = pd.concat(_dv)

    # correct units to something useful
    df['Ids (A)'] = df['Ids (A)'] * 1000
    df['Error (A)'] = df['Error (A)'] * 1000
    df['Current (A)'] = df['Current (A)'] * 1000
    setpointscale = False
    if np.where(np.abs(df['Setpoint'].values) < 1e-6)[0].any():  # quick way to see if voltages or current
        setpointscale = True
        df['Setpoint'] = df['Setpoint'] * 1e9  # nA

    df.index.rename('Time (ms)', inplace=True)
    df.rename(columns={'Ids (A)': 'Ids (mA)', 'Error (A)': 'Error (mA)',
                       'Current (A)': 'Current (mA)'}, inplace=True)
    if setpointscale:
        df.rename(columns={'Setpoint': 'Setpoint Current (nA)'}, inplace=True)
    else:
        df.rename(columns={'Setpoint': 'Setpoint Voltage (V)'}, inplace=True)

    # Add the setpoints as attributes for easy reference
    try:
        df.setpoints = pd.unique(df['Setpoint Current (nA)'])
        df.is_cc = True  # constant current
        df.is_cv = False
    except:
        df.setpoints = pd.unique(df['Setpoint Voltage (V)'])
        df.is_cc = False  # constant current
        df.is_cv = True

    df.name = path.split(r'/')[-1][:-4]

    return df, device


def plot_ccurrent(df, v_comp=-0.9):
    '''
    Only plots the current where voltage doesn't saturate
    
    This only matters if you don't crop the voltage upon reading it in
    
    v_comp : float
        The voltage compliance limit during constant_current traces
    '''

    fig, ax = plt.subplots(figsize=(9, 6))

    s = 'Setpoint Current (nA)' if df.is_cc == True else 'Setpoint Voltage (V)'

    for i in df.setpoints:
        d = df.loc[df[s] == i]
        yy = d.loc[np.abs(d['Voltage (V)']) <= np.abs(v_comp)]
        xx = d.loc[np.abs(d['Voltage (V)']) <= np.abs(v_comp)].index.values
        ax.plot(xx, yy['Ids (mA)'])

    ax.legend(labels=df.setpoints)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (mA)')
    ax.set_title(df.name)
    plt.tight_layout()

    return


def plot_current(df, norm=False, log=False):
    fig, ax = plt.subplots(figsize=(9, 6))

    s = 'Setpoint Current (nA)' if df.is_cc == True else 'Setpoint Voltage (V)'

    for i in df.setpoints:
        if norm:
            xx = df.loc[df[s] == i]['Ids (mA)'].index.values
            yy = df.loc[df[s] == i]['Ids (mA)'].values
            yy = (yy - np.min(yy)) / (np.max(yy) - np.min(yy))

            ax.plot(xx, yy)
        else:
            if log:
                ax.plot(np.abs(df.loc[df[s] == i]['Ids (mA)']))
                ax.set_yscale('log')
            else:
                ax.plot(df.loc[df[s] == i]['Ids (mA)'])
    if df.is_cc:
        ax.legend(labels=[str(x) + ' nA' for x in df.setpoints])
    else:
        ax.legend(labels=[str(x) + ' V' for x in df.setpoints])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (mA)')
    ax.set_title(df.name)
    plt.tight_layout()

    return ax


def friedlein_decay(t, mu, Cd, Cs, L, Vg, Rs, Vt, Vd, Ierr):
    '''
    Modified version of the Friedlein model taking into account an exponential
    plateau for mobility to account for slow ion uptake (carrier-dependent mobility)
    
    Uses two separate capacitances to fit the Ids response
        Cd = gate-drain capacitance
        Cd = gate-source capacitance
        Rs = electrolyte resistance
        Vg = gate voltage (should be constant)
        Vt = threshold voltage (should be constant)
        Vd = drain voltage (should be constant)
        L = channel length (should be constant, in cm)
        mu = mobility (1e-8 to 10 cm^2/V*s is reasonable)
        Ierr = current error (y-offset)
        f = percentage scaling for current into Drain vs Source
    
    '''
    f = 0.5
    C = (f * Cd + (1 - f) * Cs) / f

    #    C = Cd + Cs
    tau = Rs * C
    Vch = Vg * (1 - np.exp(-t / tau))

    p = (1 - np.exp(-t / tau)) * (0.1 / 0.025 - 1)  # from 0 to ~3, 0.1=disorder width, 0.025 = kT/q
    K = (C / L ** 2) * mu  # represents increase in density, here over ~3 o.o.m.

    # Vt0, _ = getVt(Vt, K, Vch, Vd)

    Ids = K * (Vch - Vt - Vd / 2) * Vd + Ierr

    return Ids


def model_friedlein(device, index=-0.8, multi=True, params=None):
    '''
    Wrapper for generating a friedlein_multi fit
    '''
    if multi:
        fmodel = lmfit.Model(friedlein_multi)
    else:
        fmodel = lmfit.Model(friedlein_decay)

    if not any([params]):
        params = fmParams(fmodel)

    t = device[index].index
    Ids = np.abs(device[index]['Ids (A)'])  # easier to track everything as positive numbers

    # pre-condition the Vt range
    value = getVt(params['Vt'], *preVt(params, t), params['Vd'])[0]
    params['Vt'].set(min=value * 0.3, max=value * 1.7, value=value)
    print(params['Vt'])

    result = fmodel.fit(params=params, t=t, data=Ids, method='powell')

    # feeds first run into a second run
    params = result.params
    params['Vt'].set(value=getVt(result.params['Vt'], *preVt(params, t), params['Vd'])[0])
    #    print(params['Vt'])
    #
    result = fmodel.fit(params=params, t=t, data=Ids, method='powell')
    print(result.fit_report())
    p = result.params.valuesdict()

    C = p['Cd'] + p['Cs']
    tau = p['Rs'] * C

    print('tau= ', tau, ' s')
    result.plot(xlabel='Time (s)', ylabel='Ids (A)')
    plt.tight_layout()

    return fmodel, result


def friedlein_multi(t, mu, Cd, Cs, L, Vg, Rs, Vt, Vd, Ierr):
    '''
    Modified version of the Friedlein model taking into account that we move
    from saturation to linear regime during the gate voltage pulse
    
    Uses two separate capacitances to fit the Ids response
        Cd = gate-drain capacitance
        Cs = gate-source capacitance
        Rs = electrolyte resistance
        Vg = gate voltage (should be constant)
        Vt = threshold voltage (should be constant)
        Vd = drain voltage (should be constant)
        L = channel length (should be constant, in cm)
        mu = mobility (1e-8 to 10 cm^2/V*s is reasonable)
        Ierr = current error (y-offset)
    
    '''
    #    C = Cd + Cs
    C = Cd + Cs
    K = mu * C / L ** 2
    tau = Rs * C

    Vch = Vg * (1 - np.exp(-t / tau))
    Ids = np.zeros(len(t))
    regime = []

    sat = 0
    lin = 0

    # For a given device, need Vt such that regimes meet.
    #    Vt0, _ = getVt(Vt, K, Vch, Vd)
    #    print('Vt', Vt0)
    Vt0 = Vt

    for tm, x in zip(t, range(len(Ids))):

        Vch = Vg * (1 - np.exp(-tm / tau))
        # scale mobility with empirical carrier-dependent factor
        p = (1 - np.exp(-tm / tau)) * (0.05 / 0.025 - 1)  # from 0 to ~3, 0.1=disorder width, 0.025 = kT/q

        #        K = (C/L**2) * (1 - np.exp(-tm/tau))* mu   # represents increase in density
        K = (C / L ** 2) * mu
        # saturation
        if Vch > Vt0 and Vd >= Vch:

            # print('a')
            regime.append('sat')
            sat += 1

            Ids[x] = 0.5 * K * (Vch - Vt0) ** 2 + Ierr

        # linear
        elif Vch > Vt0 and Vd < Vch:

            # print('b')
            regime.append('lin')
            lin += 1

            Ids[x] = K * (Vch - Vt0 - Vd / 2) * Vd + Ierr

            # subthreshold
        else:

            regime.append('sub')
            Ids[x] = 0
            Ids[x] = 0.5 * K * (Vch - Vt0) ** 2 + Ierr

    #    print('sat', sat,'; lin', lin)
    #    print(regime)
    return Ids


def vtdiff(Vt, K, Vch, Vd):
    return (0.5 * K * (Vch - Vt) ** 2) - (K * (Vch - Vt - Vd / 2) * Vd)


def getVt(Vt, K, Vch, Vd):
    '''
    Uses root solver to find minimum of difference between saturation line
     and linear regime line (when they intersect)
    Does this for all values of Vch given the slow ionic charging
    The optimal threshold voltage defining the overlap is then extracted
    By default the plateau of Vch should be the right Vt, which is roots[-1]
    
    Parameters:
        Vt : float
            Initial Vt guess
        K : float
            K = mu * (Cd + Cs)/L**2
        Vch : list or array
            A list of channel-gate voltages (after electrolyte)
        Vd : float
            drain voltage, should be fixed
    
    Returns:
        roots : list
            All the roots at each time step 
        roots[-1] : float
            The final Vt 
    '''
    roots = []
    for v in Vch:
        root = fsolve(vtdiff, Vt, args=(K, v, Vd))
        roots.append(root[0])

    return roots[-1], roots


def preVt(params, t):
    p = params.valuesdict()

    C = p['Cd'] + p['Cs']
    K = p['mu'] * C / p['L'] ** 2
    tau = p['Rs'] * C
    Vch = p['Vg'] * (1 - np.exp(-t / tau))

    return K, Vch


def fmParams(model):
    params = model.make_params(mu=1e-5, Cd=1e-2, Cs=1e-2, L=20e-4, Vg=0.85,
                               Vt=0.25, Vd=0.6, Ierr=0, Rs=1000)

    params['mu'].set(min=1e-8, max=100)
    params['Cd'].set(min=0)
    params['Cs'].set(min=0)
    params['L'].set(vary=False)
    params['Vg'].set(vary=False)
    params['Vd'].set(vary=False)
    params['Vt'].set(min=0.0, max=0.9)
    params['Rs'].set(min=500)
    #    params['f'].set(min=0, max=1)

    return params


'''
FARIA MODEL
'''


def fit_faria(device, key=-0.8):
    famodel = lmfit.Model(faria)
    params = famodel.make_params(I0=0, V0=-0.85, gm=1e-3, Rd=1000, Rs=100,
                                 Cd=1, f=0.5)

    # set up key params
    params['V0'].vary = False
    params['gm'].set(min=1e-6, max=100)
    params['Rd'].set(min=0.01)
    params['Rs'].set(min=0.01)
    params['Cd'].set(min=1e-15, max=1)
    params['f'].set(min=0, max=1)

    '''
    fit, return result. 
    params = result.params for fit parameters
    
    To visualize:
        print(result.fit_report())
        result.plot()
    '''
    result = famodel.fit(params=params, t=device[-0.7].index,
                         data=device[key]['Ids (A)'])
    print(result.fit_report())
    result.plot()

    return famodel, result


def faria(t, I0, V0, gm, Rd, Rs, Cd, f):
    '''
    Faria Org Elec model
    Organic Electronics 45, pp. 215-221 (2015)
    '''

    Ig = V0 * (gm * Rd - f) / (Rd + Rs)
    Ich = V0 * Rd * (gm * Rs + f) / (Rs * (Rd + Rs)) * np.exp(-t * (Rd + Rs) / (Cd * Rd * Rs))

    return I0 + Ig - Ich


# Fitting functions

def line_f(x, a, b):
    return a + b * x


'''
BERNARDS
'''


def bernards(t, del_I, f, tau_e, tau_i, y_err):
    '''
    Bernards model fitting
    Adv. Funct. Mater. 17, pp. 3538–3544 (2007)
    
    Note that this is the constant gate voltage model
    The constant gate current model is basically a line, which is only valid
     for very low gate currents (because voltage compliance)
    
    '''
    return y_err + del_I * (1 - f * tau_e / tau_i) * np.exp(-t / tau_i)


def friedlein(t, mu, Cg, L, Vg, Rg, Vt, Vd):
    '''
    Friedlein transient model
    Adv. Mater. 28, pp. 8398–8404 (2016)
    
    Assumes constant voltage step, not current
    
    In linear regime for Ids
    '''

    return (mu * Cg / L ** 2) * (Vt - Vg * -np.expm1(-t / (Rg * Cg)) - Vd / 2) * Vd


def friedlein_sat(t, mu, Cg, L, Vg, Rg, Vt, Ierr):
    '''
    Friedlein transient model
    Adv. Mater. 28, pp. 8398–8404 (2016)
    
    Assumes constant voltage step, not current
    
    In saturation regime for Ids
    '''

    return (mu * Cg / L ** 2) * (Vg * -np.expm1(-t / (Rg * Cg)) - Vt) ** 2 + Ierr


def fit_time(df, func='bernards', plot=True):
    xx = df.index.values / 1000.0

    yy = df.values[:, 0]

    # Bernards model parameters
    y_err = 0
    del_I = -np.min(yy)  # change in drain current
    tau_e = 1e-5  # electronic response time
    tau_i = 1e-1  # ionic diffusion time

    # Friedlein model parameters
    mu = 1e-2
    L = 20e-6
    Cg = 1  # "ionic" capacitance, around 100 nF
    Vt = -0.4
    Vd = -0.6
    Rg = 1e3  # ionic resistance,

    # Faria model
    I0 = yy[0]  # initial current
    V0 = -0.85  # gate voltage
    gm = 1e-3  # 1 mS
    Rd = 100e3  # 1 kOhm, channel resistance
    Cd = 100e-3  # channel capacitance
    Rs = 2e3  # solution resistance
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


# older data manipulation analysis

def find_turnon(df, current=-1e-7):
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
        f = int(np.floor(df.loc[df['Setpoint'] == i].index.values[mx] / 10000)) * 10000
        if f == 0:
            f = 10000
        xx = df.loc[df['Setpoint'] == i].loc[f:].index.values
        yy = df.loc[df['Setpoint'] == i]['Ids (A)'].loc[f:].values
        d[i] = yy
        d = d.set_index(xx - xx[0])
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
        d = d.set_index(xx - xx[0])
        device[i] = d

    df_total = pd.concat([device[a] for a in device])
    df_total.currents = df.currents

    return df_total, device

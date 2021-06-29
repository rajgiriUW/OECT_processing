# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:15:37 2019

@author: GingerLab
"""

import lmfit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve


def read_time_dep(path, start=0, stop=0, v_limit=None, skipfooter=1):
    '''
    Reads in the time-dependent data using Raj's automated version
    Saves all the different current 
    
    start : int
        Time index (in ms) to set as t=0 for the data
    stop : int
        Time index (in ms) to set as the end point for the data
    v_limit : float
        Limits constant current data to when the voltage_compliance limit is reached
        Do not use this for Constant Voltage data
        e.g. -0.9 V is typical
        
    Returns:
        
    df : DataFrame
        Single DataFrame with all the indices corrected
    
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
    norm = 1 + (ids - np.min(np.abs(ids)))/ (np.max(np.abs(ids)) - np.min(np.abs(ids)))
    df['I_DS_norm (A)'] = norm

    return df


def plot_current(df, v_comp=-1, norm= False, plot_voltage=True):
    '''
    Plot the current data
    Only plots the current where voltage doesn't saturate  (v_comp)
 
    v_comp : float
        The voltage compliance limit during constant_current traces
        This has no real effect in a constant_voltage case
        
    norm : bool, optional
        Normalize the current
        
    plot_voltage : bool, optional
        Plot the voltage (only useful for constant voltage plotting)
    '''

    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')

    if df.is_cc:
        yy = df.loc[np.abs(df['V_G (V)']) <= np.abs(v_comp)]
        xx = df.loc[np.abs(df['V_G (V)']) <= np.abs(v_comp)].index.values
    else:
        yy = df
        xx = df.index.values
       
    if norm:
        ax.plot(xx / 1000, yy['I_DS_norm (A)'], 'b')
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
    
    return y0 + A * np.exp(-t/tau)


def fit_cycles(df, doping_time, dedoping_time, func = expf, norm=False, 
               plot=True, p_type = True, dope_tau_p0 = 10, dedope_tau_p0 = 1):
    '''
    Simple single exponential fitting of kinetics
    
    df : Pandas DataFrame
        The DataFrame containing our kinetics information
        
    doping_time : int
        The time (seconds) the sample is doped
        
    dedoping_time : int
        The time (seconds) the sample is dedoped
    
    func : function
        default = single exponential. The fitting function to use. 
        Currently this is fixed for dopnig and dedoping
    
    norm : bool optional
        If True, normalizes the current data prior to fitting
        Sometimes needed for very small currents
    
    plot : bool, optional
        Displays the current and then overlays the fit per cycle
        
    p_type: bool, optional
        Assumes doping is a negative voltage
    
    dope_tau_p0 : float, optional
        The initial guess for the doping tau
        
    dedope_tau_p0 : float, optional
        The initial guess for the dedoping tau
    '''
    
    t = df.index.values / 1000 # in seconds
    
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
        
        print(n)
        _dope = range(doping_idx[n],dedoping_idx[n]+1)
        
        try:
            _dedope = range(dedoping_idx[n],doping_idx[n+1]+1)
        except:
            _dedope = range(dedoping_idx[n],len(t))
        # X-axis times
        xx_dope = t[_dope]
        xx_dedope = t[_dedope]
        
        if norm:
            yy_dope = df['I_DS_norm (A)'].iloc[_dope]
            yy_dedope = df['I_DS_norm (A)'].iloc[_dedope]
        else:
            yy_dope = df['I_DS (A)'].iloc[_dope]
            yy_dedope = df['I_DS (A)'].iloc[_dedope]
        
        # Curve fitting
        popt_dope,_ = curve_fit(func, xx_dope-xx_dope[0], yy_dope,
                   p0=[yy_dope.iloc[0],
                       yy_dope.max() - yy_dope.min(), 
                       dope_tau_p0])
        popt_dedope,_ = curve_fit(func, xx_dedope-xx_dedope[0], yy_dedope,
                   p0=[yy_dedope.iloc[0], 
                       -(yy_dedope.max() - yy_dedope.min()), 
                       dedope_tau_p0])
        doping_fits.append(popt_dope)
        dedoping_fits.append(popt_dope)
        
        if plot:
            
            ax.plot(xx_dope, func(xx_dope-xx_dope[0], *popt_dope), 'r--')
            ax.plot(xx_dedope, func(xx_dedope-xx_dedope[0], *popt_dedope), 'g--')
    
    return doping_fits, dedoping_fits


#### Model fitting ####
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


def bernards_cc(t, Ig, f, tau_e, tau_i, i_ss):
    '''
    Bernards model fitting
    Adv. Funct. Mater. 17, pp. 3538–3544 (2007)
    
    Note that this is the constant current model, which typically is only
    valid at low gate currents becaues of voltage compliance
    
    '''
    return i_ss - Ig * (f + t / tau_e)


def bernards_cv(t, del_I, f, tau_e, tau_i, i_ss):
    '''
    Bernards model fitting
    Adv. Funct. Mater. 17, pp. 3538–3544 (2007)
    
    Note that this is the constant gate voltage model
    The constant gate current model is basically a line, which is only valid
     for very low gate currents (because voltage compliance)
    
    '''
    return i_ss + del_I * (1 - f * tau_e / tau_i) * np.exp(-t / tau_i)


def lmfit_bernards(df, v_d=-0.6, Ig=1e-6):
    '''
    For Bernards fitting of constant CURRENT data
    df : pandas DataFrame
    v_d : float, optional
        The drain voltage used during this run
    '''
    xx = (df.index.values - df.index.values[0]) / 1000.0
    yy = df['I_DS(A)'].values * 1e6  # to get into uA instead of A

    bmod = lmfit.Model(bernards_cc)
    i_ss = yy[0]
    del_I = -np.min(yy)  # change in drain current
    tau_e = 1e-5  # electronic response time
    tau_i = 1  # ionic diffusion time
    L = 20e-4  # channel length, fixed, 20 um= 20e-4 cm
    params = bmod.make_params(Ig=Ig, f=0.5, tau_e=tau_e,
                              tau_i=tau_i, i_ss=i_ss)

    # set up key params
    params['f'].vary = False
    params['tau_e'].set(min=1e-8, max=1e-2)
    params['tau_i'].set(min=1e-8, max=1e3)
    params['i_ss'].set(min=np.min(yy), max=np.max(yy))
    params['Ig'].vary = False
    # params['del_I'].set()

    result = bmod.fit(params=params, t=xx, data=yy)
    print(result.fit_report())
    result.plot()

    tau_e = result.values['tau_e']
    del_I = result.values['Ig']
    # tau_i = result.values['tau_i']

    mob = np.abs(L ** 2 / (tau_e * v_d))
    slope = del_I / tau_e
    print('mobility =', mob, 'cm^2/V-s')
    print('dIsd/dIg =', del_I / tau_e, 'uA/s')

    return bmod, result, mob, slope


'''
FRIEDLEIN (CONSTANT VOLTAGE STEP)
'''


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

    yy = df['I_DS(A)'].values

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
        popt, _ = curve_fit(bernards_cv, xx, yy, p0=[del_I, 0.5, tau_e, tau_i, y_err])
    elif func is 'friedlein':
        popt, _ = curve_fit(friedlein, xx, yy, p0=[mu, Cg, L, -0.8, Rg, Vt, Vd])
    elif func is 'faria':
        popt, _ = curve_fit(faria, xx, yy, p0=[I0, V0, gm, Rd, Rs, Cd, f])

    if plot:
        plt.figure()
        plt.plot(xx, yy, 'b-', linewidth=3)

        if func is 'bernards':
            plt.plot(xx, bernards_cv(xx, *popt), 'r--', linewidth=3)
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

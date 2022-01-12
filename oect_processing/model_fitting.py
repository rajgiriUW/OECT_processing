# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 19:16:38 2021

@author: Raj
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve


#### Model fitting ####
def friedlein_decay(t, mu, Cd, Cs, L, Vg, Rs, Vt, Vd, Ierr):
    '''
    Modified version of the Friedlein model taking into account an exponential
    plateau for mobility to account for slow ion uptake (carrier-dependent mobility)
    
    Uses two separate capacitances to fit the Ids response
        
    :param Cd: gate-drain capacitance
    :type Cd: float
    
    :param Cs: gate-source capacitance
    :type Cs: float
    
    :param Rs: electrolyte resistance
    :type Rs: float
    
    :param Vg: gate voltage (should be constant)
    :type Vg: float
    
    :param Vt: threshold voltage (should be constant)
    :type Vt: float
    
    :param Vd: drain voltage (should be constant)
    :type Vd: float
    
    :param L: channel length (should be constant, in cm)
    :type L: float
    
    :param mu: mobility (1e-8 to 10 cm^2/V*s is reasonable)
    :type mu: float
    
    :param Ierr: current error (y-offset)
    :type Ierr: float
    
    :param f: percentage scaling for current into Drain vs Source
    :type f: float
    
    :returns: [describe return value here]
    :rtype:
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
    
    :param device:
    :type device:
    
    :param index:
    :type index: float
    
    :param multi:
    :type multi: bool
    
    :param params:
    :type params:
    
    :returns: tuple (fmodel, result)
        WHERE
        [type] fmodel is...
        [type] result is...
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
    
    :param Cd: gate-drain capacitance
    :type Cd: float
    
    :param Cs: gate-source capacitance
    :type Cs: float
    
    :param Rs: electrolyte resistance
    :type Rs: float
    
    :param Vg: gate voltage (should be constant)
    :type Vg: float
    
    :param Vt: threshold voltage (should be constant)
    :type Vt: float
    
    :param Vd: drain voltage (should be constant)
    :type Vd: float
    
    :param L: channel length (should be constant, in cm)
    :type L: float
    
    :param mu: mobility (1e-8 to 10 cm^2/V*s is reasonable)
    :type mu: float
    
    :param Ierr: current error (y-offset)
    :type Ierr: float
    
    :returns:
    :rtype:
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
    '''
    :param Vt:
    :type Vt:
    
    :param K:
    :type K:
    
    :param Vch:
    :type Vch:
    
    :param Vd:
    :type Vd:
    
    :returns:
    :rtype:
    '''

    return (0.5 * K * (Vch - Vt) ** 2) - (K * (Vch - Vt - Vd / 2) * Vd)


def getVt(Vt, K, Vch, Vd):
    '''
    Uses root solver to find minimum of difference between saturation line
    and linear regime line (when they intersect)
    Does this for all values of Vch given the slow ionic charging
    The optimal threshold voltage defining the overlap is then extracted
    By default the plateau of Vch should be the right Vt, which is roots[-1]
    
    :param Vt: Initial Vt guess
    :type Vt: float
    
    :param K: K = mu * (Cd + Cs)/L**2
    :type K: float
    
    :param Vch: A list of channel-gate voltages (after electrolyte)
    :type Vch: list or array
    
    :param Vd: drain voltage, should be fixed
    :type Vd: float

    :returns: tuple (roots, roots[-1])
        WHERE
        list roots is All the roots at each time step 
        float roots[-1] is the final Vt 
    '''
    roots = []
    for v in Vch:
        root = fsolve(vtdiff, Vt, args=(K, v, Vd))
        roots.append(root[0])

    return roots[-1], roots


def preVt(params, t):
    '''
    :param params:
    :type params:
    
    :param t:
    :type t: float
    
    :returns: tuple (K, Vch)
        WHERE
        [type] K is ...
        [type] Vch is ...
    '''
    p = params.valuesdict()

    C = p['Cd'] + p['Cs']
    K = p['mu'] * C / p['L'] ** 2
    tau = p['Rs'] * C
    Vch = p['Vg'] * (1 - np.exp(-t / tau))

    return K, Vch


def fmParams(model):
    '''
    :param model:
    :type model:
    
    :returns:
    :rtype:
    '''
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
    '''
    :param device:
    :type device:
    
    :param key:
    :type key: float
    
    :returns: tuple (famodel, result)
        WHERE
        [type] famodel is ...
        [type] result is ...
    '''
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
    
    :param t:
    :type t:
    
    :param I0:
    :type I0:
    
    :param V0:
    :type V0:
    
    :param gm:
    :type gm:
    
    :param Rd:
    :type Rd:
    
    :param Rs:
    :type Rs:
    
    :param Cd:
    :type Cd:
    
    :param f:
    :type f:
    
    :returns:
    :rtype:
    '''

    Ig = V0 * (gm * Rd - f) / (Rd + Rs)
    Ich = V0 * Rd * (gm * Rs + f) / (Rs * (Rd + Rs)) * np.exp(-t * (Rd + Rs) / (Cd * Rd * Rs))

    return I0 + Ig - Ich


# Fitting functions

def line_f(x, a, b):
    '''
    :param x:
    :type x:
    
    :param a:
    :type a:
    
    :param b:
    :type b:
    
    :returns:
    :rtype:
    '''
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
    
    :param t:
    :type t:
    
    :param Ig:
    :type Ig:
    
    :param f:
    :type f:
    
    :param tau_e:
    :type tau_e:
    
    :param tau_i:
    :type tau_i:
    
    :param i_ss:
    :type i_ss:
    
    :returns:
    :rtype:
    '''
    return i_ss - Ig * (f + t / tau_e)


def bernards_cv(t, del_I, f, tau_e, tau_i, i_ss):
    '''
    Bernards model fitting
    Adv. Funct. Mater. 17, pp. 3538–3544 (2007)
    
    Note that this is the constant gate voltage model
    The constant gate current model is basically a line, which is only valid
     for very low gate currents (because voltage compliance)
    
    :param t:
    :type t:
    
    :param del_I:
    :type del_I:
    
    :param f:
    :type f:
    
    :param tau_e:
    :type tau_e:
    
    :param tau_i:
    :type tau_i:
    
    :param i_ss:
    :type i_ss:
    
    :returns:
    :rtype:
    '''
    return i_ss + del_I * (1 - f * tau_e / tau_i) * np.exp(-t / tau_i)


def lmfit_bernards(df, v_d=-0.6, Ig=1e-6):
    '''
    For Bernards fitting of constant CURRENT data
    
    :param df:
    :type df: pandas DataFrame
    
    :param v_d: The drain voltage used during this run
    :type v_d: float
    
    :param Ig:
    :type Ig: float
    
    :returns: tuple (bmod, result, mob, slope)
        WHERE
        [type] bmod is...
        [type] result is...
        [type] mob is...
        [type] slope is...
        
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
    
    :param t:
    :type t:
    
    :param mu:
    :type mu:
    
    :param Cg:
    :type Cg:
    
    :param L:
    :type L:
    
    :param Vg:
    :type Vg:
    
    :param Rg:
    :type Rg:
    
    :param Vt:
    :type Vt:
    
    :param Vd:
    :type Vd:
    
    :returns:
    :rtype:
    '''

    return (mu * Cg / L ** 2) * (Vt - Vg * -np.expm1(-t / (Rg * Cg)) - Vd / 2) * Vd


def friedlein_sat(t, mu, Cg, L, Vg, Rg, Vt, Ierr):
    '''
    Friedlein transient model
    Adv. Mater. 28, pp. 8398–8404 (2016)
    
    Assumes constant voltage step, not current
    
    In saturation regime for Ids
    
    :param t:
    :type t:
    
    :param mu:
    :type mu:
    
    :param Cg:
    :type Cg:
    
    :param L:
    :type L:
    
    :param Vg:
    :type Vg:
    
    :param Rg:
    :type Rg:
    
    :param Vt:
    :type Vt:
    
    :param Ierr:
    :type Ierr:
    
    :returns:
    :rtype:
    '''

    return (mu * Cg / L ** 2) * (Vg * -np.expm1(-t / (Rg * Cg)) - Vt) ** 2 + Ierr


def fit_time(df, func='bernards', plot=True):
    '''
    :param df:
    :type df:
    
    :param func:
    :type func: str
    
    :param plot:
    :type plot: bool
    
    :returns:
    :rtype:
    '''
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
    '''
    :param df:
    :type df:
    
    :param current:
    :type current: float
    
    :returns: tuple (mx, npts)
        WHERE
        [type] mx is...
        [type] npts is...
    '''
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
    Crops all the data before the initial turn-on event. Manually shifts to 
    nearest 10000 ms point (assumes I set at an even second mark)
    df_total, device = timedep.crop_prepulse(df)
    
    :param df: dataframe from read_time_dep
    :type df: dataframe
    
    :returns: tuple (df_total, device)
        WHERE
        DataFrame df_total is the big dataframe with all the data (doesn't standardize times)
        dict device is the dictionary of currents
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
    Crops all the data before the initial turn-on event. Manually shifts to 
    nearest 10000 ms point (assumes I set at an even second mark)
    df_total, device = timedep.crop_prepulse(df)
    
    :param df: dataframe from read_time_dep
    :type df: dataframe
    
    :param timeon:
    :type timeon: float
    
    :returns: tuple (df_total, device)
        WHERE
        DataFrame df_total is the big dataframe with all the data (doesn't standardize times)
        dict device is the dictionary of currents
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

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
    Modified Friedlein model with exponential mobility plateau for slow ion uptake.

    Uses two separate capacitances (gate-drain and gate-source) to fit the Ids response.

    Parameters
    ----------
    t : array-like
        Time values.
    mu : float
        Mobility in cm^2/V*s (reasonable range: 1e-8 to 10).
    Cd : float
        Gate-drain capacitance (F).
    Cs : float
        Gate-source capacitance (F).
    L : float
        Channel length in cm.
    Vg : float
        Gate voltage (constant).
    Rs : float
        Electrolyte resistance (Ohm).
    Vt : float
        Threshold voltage (constant).
    Vd : float
        Drain voltage (constant).
    Ierr : float
        Current offset (y-intercept error).

    Returns
    -------
    ndarray
        Modelled drain current Ids(t).
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
    Wrapper for fitting the Friedlein transient model to device data.

    Parameters
    ----------
    device : dict
        Dict of DataFrames keyed by gate voltage, each containing 'Ids (A)'.
    index : float, optional
        Gate voltage key to use for fitting.
    multi : bool, optional
        If True, uses friedlein_multi (saturation + linear regime); otherwise friedlein_decay.
    params : lmfit.Parameters, optional
        Initial fit parameters. Defaults are generated via fmParams if not provided.

    Returns
    -------
    fmodel : lmfit.Model
    result : lmfit.ModelResult
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
    Friedlein model that accounts for transitions between saturation and linear regimes.

    Parameters
    ----------
    t : array-like
        Time values.
    mu : float
        Mobility in cm^2/V*s.
    Cd : float
        Gate-drain capacitance (F).
    Cs : float
        Gate-source capacitance (F).
    L : float
        Channel length in cm.
    Vg : float
        Gate voltage (constant).
    Rs : float
        Electrolyte resistance (Ohm).
    Vt : float
        Threshold voltage (constant).
    Vd : float
        Drain voltage (constant).
    Ierr : float
        Current offset (y-intercept error).

    Returns
    -------
    ndarray
        Modelled drain current Ids(t).
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
    Difference between saturation and linear regime currents at a given Vt.

    Parameters
    ----------
    Vt : float
        Threshold voltage guess.
    K : float
        mu * C / L^2 prefactor.
    Vch : float
        Channel voltage at a given time step.
    Vd : float
        Drain voltage.

    Returns
    -------
    float
    '''

    return (0.5 * K * (Vch - Vt) ** 2) - (K * (Vch - Vt - Vd / 2) * Vd)


def getVt(Vt, K, Vch, Vd):
    '''
    Finds the threshold voltage by root-solving the saturation/linear regime crossover.

    Iterates over all Vch values (slow ionic charging) and returns the final root,
    which corresponds to the plateau Vch and thus the correct Vt.

    Parameters
    ----------
    Vt : float
        Initial threshold voltage guess.
    K : float
        Prefactor mu * (Cd + Cs) / L^2.
    Vch : list or ndarray
        Channel-gate voltages over time (after electrolyte charging).
    Vd : float
        Drain voltage (fixed).

    Returns
    -------
    float
        Final threshold voltage (roots[-1]).
    list
        All roots at each time step.
    '''
    roots = []
    for v in Vch:
        root = fsolve(vtdiff, Vt, args=(K, v, Vd))
        roots.append(root[0])

    return roots[-1], roots


def preVt(params, t):
    '''
    Computes K and Vch(t) from lmfit parameters for use as getVt inputs.

    Parameters
    ----------
    params : lmfit.Parameters
        Must contain mu, Cd, Cs, L, Rs, Vg.
    t : array-like
        Time values.

    Returns
    -------
    K : float
        mu * C / L^2 prefactor.
    Vch : ndarray
        Channel voltage as a function of time.
    '''
    p = params.valuesdict()

    C = p['Cd'] + p['Cs']
    K = p['mu'] * C / p['L'] ** 2
    tau = p['Rs'] * C
    Vch = p['Vg'] * (1 - np.exp(-t / tau))

    return K, Vch


def fmParams(model):
    '''
    Generates default lmfit Parameters for the Friedlein model.

    Parameters
    ----------
    model : lmfit.Model

    Returns
    -------
    lmfit.Parameters
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
    Fits the Faria model to device transient data.

    Parameters
    ----------
    device : dict
        Dict of DataFrames keyed by gate voltage.
    key : float, optional
        Gate voltage key to fit.

    Returns
    -------
    famodel : lmfit.Model
    result : lmfit.ModelResult
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
    Faria OECT transient model. Organic Electronics 45, pp. 215-221 (2015).

    Parameters
    ----------
    t : array-like
        Time values.
    I0 : float
        Initial current offset.
    V0 : float
        Gate voltage.
    gm : float
        Transconductance (S).
    Rd : float
        Channel resistance (Ohm).
    Rs : float
        Solution resistance (Ohm).
    Cd : float
        Channel capacitance (F).
    f : float
        Current partitioning factor (0 to 1).

    Returns
    -------
    ndarray
        Modelled drain current Ids(t).
    '''

    Ig = V0 * (gm * Rd - f) / (Rd + Rs)
    Ich = V0 * Rd * (gm * Rs + f) / (Rs * (Rd + Rs)) * np.exp(-t * (Rd + Rs) / (Cd * Rd * Rs))

    return I0 + Ig - Ich


# Fitting functions

def line_f(x, a, b):
    '''
    Linear function y = a + b*x.

    Parameters
    ----------
    x : array-like
    a : float
        Intercept.
    b : float
        Slope.

    Returns
    -------
    ndarray
    '''
    return a + b * x


'''
BERNARDS
'''


def bernards_cc(t, Ig, f, tau_e, tau_i, i_ss):
    '''
    Bernards constant-current model. Adv. Funct. Mater. 17, pp. 3538-3544 (2007).

    Valid only at low gate currents where voltage compliance is not reached.

    Parameters
    ----------
    t : array-like
        Time values.
    Ig : float
        Gate current (A).
    f : float
        Partitioning factor.
    tau_e : float
        Electronic response time (s).
    tau_i : float
        Ionic diffusion time (s).
    i_ss : float
        Steady-state drain current.

    Returns
    -------
    ndarray
    '''
    return i_ss - Ig * (f + t / tau_e)


def bernards_cv(t, del_I, f, tau_e, tau_i, i_ss):
    '''
    Bernards constant-voltage model. Adv. Funct. Mater. 17, pp. 3538-3544 (2007).

    Parameters
    ----------
    t : array-like
        Time values.
    del_I : float
        Change in drain current (A).
    f : float
        Partitioning factor.
    tau_e : float
        Electronic response time (s).
    tau_i : float
        Ionic diffusion time (s).
    i_ss : float
        Steady-state drain current.

    Returns
    -------
    ndarray
    '''
    return i_ss + del_I * (1 - f * tau_e / tau_i) * np.exp(-t / tau_i)


def lmfit_bernards(df, v_d=-0.6, Ig=1e-6):
    '''
    Fits the Bernards constant-current model using lmfit.

    Parameters
    ----------
    df : DataFrame
        Transient data with 'I_DS(A)' column and time (ms) index.
    v_d : float, optional
        Drain voltage used during the measurement.
    Ig : float, optional
        Gate current (A).

    Returns
    -------
    bmod : lmfit.Model
    result : lmfit.ModelResult
    mob : float
        Estimated mobility (cm^2/V*s).
    slope : float
        dIsd/dIg (uA/s).
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
    Friedlein transient model, linear regime. Adv. Mater. 28, pp. 8398-8404 (2016).

    Assumes a constant voltage step (not constant current).

    Parameters
    ----------
    t : array-like
        Time values.
    mu : float
        Mobility (cm^2/V*s).
    Cg : float
        Gate capacitance (F).
    L : float
        Channel length (cm).
    Vg : float
        Gate voltage (V).
    Rg : float
        Gate resistance / ionic resistance (Ohm).
    Vt : float
        Threshold voltage (V).
    Vd : float
        Drain voltage (V).

    Returns
    -------
    ndarray
        Modelled Ids(t) in linear regime.
    '''

    return (mu * Cg / L ** 2) * (Vt - Vg * -np.expm1(-t / (Rg * Cg)) - Vd / 2) * Vd


def friedlein_sat(t, mu, Cg, L, Vg, Rg, Vt, Ierr):
    '''
    Friedlein transient model, saturation regime. Adv. Mater. 28, pp. 8398-8404 (2016).

    Assumes a constant voltage step (not constant current).

    Parameters
    ----------
    t : array-like
        Time values.
    mu : float
        Mobility (cm^2/V*s).
    Cg : float
        Gate capacitance (F).
    L : float
        Channel length (cm).
    Vg : float
        Gate voltage (V).
    Rg : float
        Gate/ionic resistance (Ohm).
    Vt : float
        Threshold voltage (V).
    Ierr : float
        Current offset.

    Returns
    -------
    ndarray
        Modelled Ids(t) in saturation regime.
    '''

    return (mu * Cg / L ** 2) * (Vg * -np.expm1(-t / (Rg * Cg)) - Vt) ** 2 + Ierr


def fit_time(df, func='bernards', plot=True):
    '''
    Fits a transient current trace with a chosen model.

    Parameters
    ----------
    df : DataFrame
        Transient data with 'I_DS(A)' column and time (ms) index.
    func : str, optional
        Model to use: 'bernards', 'friedlein', or 'faria'.
    plot : bool, optional
        If True, plots the data and fit overlay.

    Returns
    -------
    ndarray
        Optimal fit parameters from curve_fit.
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
    Finds the turn-on index for a given current setpoint.

    Parameters
    ----------
    df : DataFrame
        Transient data with a 'Setpoint' column.
    current : float, optional
        Current setpoint to locate.

    Returns
    -------
    mx : int
        Index of maximum dI/dt (turn-on point).
    npts : int
        Number of points at this setpoint.
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
    Crops data before the initial turn-on event, aligning to the nearest 10000 ms mark.

    Parameters
    ----------
    df : DataFrame
        Data from read_time_dep, with a 'Setpoint' column and a 'currents' attribute.

    Returns
    -------
    df_total : DataFrame
        Concatenated data across all setpoints (times not standardised).
    device : dict
        Dict of per-setpoint DataFrames with corrected time indices.
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
    Crops data before a fixed time point, aligning all setpoints to the same start.

    Parameters
    ----------
    df : DataFrame
        Data from read_time_dep, with a 'Setpoint' column and a 'currents' attribute.
    timeon : float, optional
        Time (ms) at which to begin cropping each setpoint segment.

    Returns
    -------
    df_total : DataFrame
        Concatenated data across all setpoints (times not standardised).
    device : dict
        Dict of per-setpoint DataFrames with corrected time indices.
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

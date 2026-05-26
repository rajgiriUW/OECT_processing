# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:16:43 2019

@author: Raj
"""

import pickle

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit as cf

from .oect_utils import oect_load
from .oect_utils import oect_plot


class OECTDevice:
    '''
    Aggregates processed OECT pixels for a single device and computes uC*.

    See oect_utils.oect_load for more on uC_scale processing.

    Usage
    -----
    >>> import oect_processing as oectp
    >>> device = oectp.OECTDevice('../device_data')
    >>> device = oectp.OECTDevice('../device_data', options={'plot': [True, True]})

    Parameters
    ----------
    path : str
        Path to parent folder containing pixel subfolders '01', '02', etc.
        A config file will be auto-generated if not present.
    pixels : dict, optional
        Pre-processed pixel dict from a previous run.
    params : dict, optional
        Device parameters: d (float, film thickness in nm) or thickness (float, same).
    options : dict, optional
        spline : bool
            Use gm splines instead of the smoothed derivative.
        V_low : bool
            Detect erroneous turnover points when devices break down.
        retrace_only : bool
            Use only the retrace sweep.
        verbose : bool
            Print progress to display.
        plot : list of bool
            [0] Plot the uC* graph; [1] plot individual pixel plots.

    Attributes
    ----------
    L : float
        Channel length in microns.
    W : float
        Channel width in microns.
    d : float
        Film thickness in metres.
    WdL : ndarray
        W*d/L prefactor for each pixel (metres).
    Vg_Vt : ndarray
        Vg - Vt (gate voltage at peak gm minus threshold voltage) for each pixel.
    Vt : ndarray
        Threshold voltages.
    uC : ndarray
        uC* fit coefficients [intercept, slope] from gm vs WdL*Vg_Vt.
    uC_0 : ndarray
        uC* fit forced through the origin.
    gms : ndarray
        Peak transconductances for each pixel.
    pix_paths : list
        Folder paths for each pixel.
    pixels : dict
        OECT objects keyed by pixel folder name.
    '''

    def __init__(self,
                 path='',
                 pixels={},
                 params={},
                 options={}):

        self.path = path
        self.pixels = pixels

        if not path and not any(pixels):
            from PyQt5 import QtWidgets

            app = QtWidgets.QApplication([])
            self.path = QtWidgets.QFileDialog.getExistingDirectory(caption='Select folder of data')
            print('Loading', self.path)
            app.closeAllWindows()
            app.exit()

        self.params = {}
        for m in params:
            self.params[m] = params[m]

        self.options = {'V_low': False, 'retrace_only': False, 
                        'verbose': False, 'plot': [True, False], 
                        'spline': False}
        self.options.update(options)

        # if device has not been processed
        if not any(pixels):

            pixels, pm = oect_load.uC_scale(self.path,
                                            **self.params,
                                            **self.options)

            self.params.update(pm)
            self.pixels = pixels

        else:

            self.get_params()

        # extract a subset as direct attributes
        self.L = self.params['L']
        self.WdL = self.params['WdL']
        self.W = self.params['W']
        self.d = self.params['d']
        self.Vg_Vt = self.params['Vg_Vt']
        self.Vt = self.params['Vt']
        self.uC = self.params['uC']
        self.uC_0 = self.params['uC_0']
        self.gms = self.params['gms']

        self.pix_paths = []

        for p in self.pixels:
            self.pix_paths.append(self.pixels[p].folder)

        return

    def get_params(self):
        '''
        Generates uC* parameters from pixel data, averaging forward and backward sweeps.
        '''
        Wd_L = np.array([])
        W = np.array([])
        Vg_Vt = np.array([])  # threshold offset
        Vt = np.array([])
        gms = np.array([])

        # assumes Length and thickness are fixed
        params = {}

        for pixel in self.pixels:

            if self.pixels[pixel].gms.empty:
                self.pixels[pixel].calc_gms()
                self.pixels[pixel].thresh()

            ix = len(self.pixels[pixel].VgVts)
            Vt = np.append(Vt, self.pixels[pixel].Vts)
            Vg_Vt = np.append(Vg_Vt, self.pixels[pixel].VgVts)
            if self.options['spline'] == True:
                gms = np.append(gms, self.pixels[pixel].peak_gm_spl)
            else:
                gms = np.append(gms, self.pixels[pixel].peak_gm)
            W = np.append(W, self.pixels[pixel].W)

            # appends WdL as many times as there are transfer curves
            for i in range(len(self.pixels[pixel].VgVts)):
                Wd_L = np.append(Wd_L, self.pixels[pixel].WdL)

            # remove the trace ()
            if self.options['retrace_only'] and len(self.pixels[pixel].VgVts) > 1:
                Vt = np.delete(Vt, -ix)
                Vg_Vt = np.delete(Vg_Vt, -ix)
                gms = np.delete(gms, -ix)
                Wd_L = np.delete(Wd_L, -ix)

            params['L'] = self.pixels[pixel].L
            params['d'] = self.pixels[pixel].d

        # fit functions
        def line_f(x, a, b):

            return a + b * x

        def line_0(x, b):
            'no y-offset --> better log-log fits'
            return b * x

        # * 1e2 to get into right mobility units (cm)
        uC_0, _ = cf(line_0, Wd_L * Vg_Vt, gms)
        uC, _ = cf(line_f, Wd_L * Vg_Vt, gms)

        # Create an OECT and add arrays 
        params['WdL'] = Wd_L
        params['W'] = W
        params['Vg_Vt'] = Vg_Vt
        params['Vt'] = Vt
        params['uC'] = uC
        params['uC_0'] = uC_0
        params['gms'] = gms

        self.params = params

        self.L = self.params['L']
        self.WdL = self.params['WdL']
        self.W = self.params['W']
        self.d = self.params['d']
        self.Vg_Vt = self.params['Vg_Vt']
        self.Vt = self.params['Vt']
        self.uC = self.params['uC']
        self.uC_0 = self.params['uC_0']
        self.gms = self.params['gms']

        return

    def plot_uc(self, save=False):
        '''
        Plots the uC* scaling graph.

        Parameters
        ----------
        save : bool, optional
            If True, saves the figure to disk.
        '''
        fig = oect_plot.plot_uC(self.params, savefig=save)

        return

    def average(self, overwrite=False):
        '''
        Averages gm and Vg_Vt values across pixels at the same WdL.

        Parameters
        ----------
        overwrite : bool, optional
            If True, replaces WdL/gms/Vg_Vt in-place. Otherwise stores under self.average.
        '''

        df = pd.DataFrame(index=self.WdL)
        df['gms'] = self.gms
        df['Vg_Vt'] = self.Vg_Vt
        df = df.groupby(df.index).mean()
        if overwrite:
            self.WdL = df.index.values
            self.gms = df['gms'].values.flatten()
            self.Vg_Vt = df['Vg_Vt'].values
        else:
            self.average = {}
            self.average['WdL'] = df.index.values
            self.average['gms'] = df['gms'].values.flatten()
            self.average['Vg_Vt'] = df['Vg_Vt'].values

        return


def save(dv, append=''):
    '''
    Pickles an OECTDevice object to the device's path.

    Parameters
    ----------
    dv : OECTDevice
        Device to save.
    append : str, optional
        Label appended to the output filename.
    '''
    with open(dv.path + r'\uC_data_' + append + '.pkl', 'wb') as output:
        pickle.dump(dv, output, pickle.HIGHEST_PROTOCOL)

    return

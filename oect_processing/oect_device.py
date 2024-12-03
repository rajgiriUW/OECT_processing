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
    Class containing the processed pixels for a single device
    This simplifies comparing and plotting various uC* datasets together.
    See oect_utils.oect_load for more description on uC_scale processing
    
    Usage
    --------
    >>> import oect_processing as oectp
    >>>
    >>> path = '../device_data'
        To process the folder and not save any data (faster)
    >>> device = oectp.OECTDevice(path)
        To process the folder and save all the images
    >>> device = oectp.OECTDevice(path, options={'plot': [True, True]}) 
    where options['plot'] is defined below

    :param path: Path to parent folder containing pixels in subfolders '01', '02', etc.
        A config file will be auto-generated if not in that folder
    :type path: string
    
    :param thickness: The device layer thickness, assuming it is fixed for all dimensions
    :type thickness: float
        
    :param pixels: If passing an existing set of data from a previous run
    :type pixels: dict of oect.OECTDevice, optional

    :param params: 
        For passing specific device parameters. Currently, this only supports
        d : float
            Film thickness (nanometers)
        thickness : float
            Film thickness (nanometers)
        Both variables are the same and for ease of use (oect.OECT uses 'd')
    :type params: dict, optional
    
    :param options:
        spline: bool, optional
            Use gm splines instead of the smoothed derivative from actual data 
            This is a little experimental and interpolates voltages
        V_low : bool, optional
            Whether to find erroneous "turnover" points when devices break down
        retrace_only : bool, optional
            Only use the retrace in case trace isn't saturating
        verbose: bool, optional
            Print to display
        plot : list of bools, optional
            [0]: Plot the uC* data
            [1]: plot the individual plots
            Whether to plot or not. Plotting can be very fast if both are turned on!
    :type options: dict, optional

    Attributes
    ----------
    L : float
        Device channel lengths in MICRONS
    W : float
        Device channel widths in MICRONS
    d : float
        Device thickness in METERS
    WdL : array
        W*d/L (prefactor in gm equation) for each device, in METERS
    Vg_Vt : array
        Vg - Vt value for each device (gate voltage of peak gm minus threshold votlage)
    Vt : array
        Threshold voltage
    uC : float
        uC* extracted from the gm vs WdL * Vg_Vt plot
    uC_0 : float
        uC* forced to go through 0,0 
    gms : array
        peak transconductances for each device
    pix_paths : array
        Folder paths for the pixels
    pixels : dictionary
        Dictionary of the generated pixels using OECT class for each folder
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
                        'verbose': False, 'plot': [True, False], 'spline': False}
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
        Generates the parameters from the pixel data and calculates uC*
        By default this averages forward and backward curves together
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
        :param save:
        :type save: bool
        '''
        fig = oect_plot.plot_uC(self.params, savefig=save)

        return

    def average(self, overwrite=False):
        '''
        Averages data for the same voltages together.

        :param overwrite: If true, does not save a backup version to the class
        :type overwrite: bool, optional
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
    :param dv:
    :type dv:
    
    :param append: data label
    :type append: str
    '''
    with open(dv.path + r'\uC_data_' + append + '.pkl', 'wb') as output:
        pickle.dump(dv, output, pickle.HIGHEST_PROTOCOL)

    return

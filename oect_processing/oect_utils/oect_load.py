# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:04:47 2018

@author: Raj
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit as cf

import oect_processing as oectp

try:
    from . import oect_plot
except:
    from oect_utils import oect_plot

'''
Wrapper function for generating a uC* plot. This file contains one main function:

    load_uC : for loading the pixels to generate a uC* plot
    
Usage:
    
    >> pixels, uC_dv = oect_load.uC_scale(r'path_to_folder_of_data', thickness=40e-9, plot=[True, False)

This function assumes that folders are labeled 01, 02, etc for each pixel
The uC_Scale function then processes each subfolder of data, calculates the 
gm, Vt, and mobility (assuming you have a capacitance value)                                                          

'''


def uC_scale(path='',
             plot=[True, False],
             V_low=False,
             retrace_only=False,
             verbose=True,
             thickness=None,
             d=None,
             capacitance=None,
             c_star=None,
             params={},
             options={}):
    '''
    :param path: string path to folder '.../avg'. Note Windows path are of form r'Path_name'
    :type path: string
        
    :param thickness: approximate film thickness. Standard polymers (for Raj) are ~40 nm
    :type thickness: float
        
    :param plot:
        [0]: Plot the uC* data
        [1]: plot the individual plots
        Whether to plot or not. Not plotting is very fast!
    :type plot: list of bools, optional
    
    :param retrace_only: Only use the retrace in case trace isn't saturating
    :type retrace_only : bool, optional
    
    :param V_low: Whether to find erroneous "turnover" points when devices break down
    :type V_low : bool, optional
    
    :param verbose: print to display
    :type verbose: bool, optional
    
    :param thickness, d: The film thickness
        Both variables are the same and is for ease of use (oect.OECT uses 'd')
    :type thickness, d: float, optional
    
    :param capacitance: In Farads
        If provided, will calculate the mobility. This should be a sanity check
        against the calculated uC*, since this is somewhat circular logic
    :type capacitance: float, optional
        
    :param c_star: in Farad / cm^3 NOTE THE CENTIMETERS^3 units
        This value is calculated from EIS or so
    :type c_star: float, optional
    
    :param params:
    :type params: dict
    
    :param options:
    :type options: dict
    
    :returns: tuple (pixels, uC_dv)
        WHERE
        dict pixels contains the various OECT class devices
        dict uC_dv contains
            Wd_L : ndarray
                coefficient for plotting on x-axis
            
            gms : ndarray
                average transconductance for plotting on y-axis
            
            Vg_Vt : ndarray
                threshold voltage shifts for correcting uC* fit
    
    '''
    if not path:
        path = file_open(caption='Select uC subfolder')
        print('Loading from', path)

    filelist = os.listdir(path)

    f = filelist[:]
    for k in filelist:
        try:
            sub_num = int(k)
        except:
            if verbose:
                print('Ignoring', k)
            f.remove(k)
    filelist = f[:]
    paths = [os.path.join(path, name) for name in filelist]
    pixkeys = [f + '_uC' for f in filelist]

    # removes random files instead of the sub-folders
    for p in paths:
        if not os.path.isdir(p):
            paths.remove(p)

    pixels = {}
    opts = {'V_low': V_low}
    if any(options):
        for o in options:
            opts[o] = options[o]
    if thickness:
        d = thickness
    for k, v in {'d': d, 'capacitance': capacitance, 'c_star': c_star}.items():
        if v:
            params[k] = v

    # loads all the folders
    if type(plot) == bool or len(plot) == 1:
        plot = [plot, plot]

    for p, f in zip(paths, pixkeys):

        if os.listdir(p):

            if verbose:
                print(p)
            print(params)
            dv = loadOECT(p, params, gm_plot=plot, plot=plot[1], options=opts, verbose=verbose)
            pixels[f] = dv

        else:

            pixkeys.remove(f)

    # do uC* graphs, need gm vs W*d/L
    Wd_L = np.array([])
    W = np.array([])
    Vg_Vt = np.array([])  # threshold offset
    Vt = np.array([])
    gms = np.array([])
    mobility = np.array([])

    # assumes Length and thickness are fixed
    uC_dv = {}

    for pixel in pixels:

        if not pixels[pixel].gms.empty:

            ix = len(pixels[pixel].VgVts)
            Vt = np.append(Vt, pixels[pixel].Vts)
            Vg_Vt = np.append(Vg_Vt, pixels[pixel].VgVts)
            gms = np.append(gms, pixels[pixel].gm_peaks['peak gm (S)'].values)
            W = np.append(W, pixels[pixel].W)
            mobility = np.append(mobility, pixels[pixel].mobility)

            # appends WdL as many times as there are transfer curves
            for i in range(len(pixels[pixel].VgVts)):
                Wd_L = np.append(Wd_L, pixels[pixel].WdL)
            # remove the trace ()
            if retrace_only and len(pixels[pixel].VgVts) > 1:
                Vt = np.delete(Vt, -ix)
                Vg_Vt = np.delete(Vg_Vt, -ix)
                gms = np.delete(gms, -ix)
                Wd_L = np.delete(Wd_L, -ix)

            uC_dv['L'] = pixels[pixel].L
            uC_dv['d'] = pixels[pixel].d

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
    uC_dv['WdL'] = Wd_L
    uC_dv['W'] = W
    uC_dv['Vg_Vt'] = Vg_Vt
    uC_dv['Vt'] = Vt
    uC_dv['uC'] = uC
    uC_dv['uC_0'] = uC_0
    uC_dv['gms'] = gms
    uC_dv['folder'] = path
    uC_dv['mobility'] = mobility

    if plot[0]:
        fig = oect_plot.plot_uC(uC_dv)
        fig = oect_plot.plot_uC(uC_dv, average=True, label='avg')

        if verbose:
            print('uC* = ', str(uC_0 * 1e-2), ' F/cm*V*s')

    if verbose:
        print('Vt = ', uC_dv['Vt'])

    return pixels, uC_dv


def loadOECT(path, params=None, gm_plot=True, plot=True, options={}, verbose=True):
    """
    Wrapper function for processing OECT data
    USAGE:
        device1 = loadOECT(folder_name)
        
    :param path:
    :type path: str
    
    :param params: params = {W: , L: , d: } for W, L, d of device
    :type params: dict
    
    :param gm_plot:
    :type gm_plot: bool
    
    :param plot:
    :type plot: bool
    
    :param options:
    :type options: dict
    
    :param verbose:
    :type verbose: bool
    
    :returns:
    :rtype:
    """

    if not path:
        path = file_open(caption='Select device subfolder')

    device = oectp.OECT(path, params=params, options=options)
    device.calc_gms()
    device.thresh()

    scaling = device.WdL  # W *d / L

    if verbose:
        for key in device.gms:
            print(key, ': {:.2f}'.format(np.max(device.gms[key].values * 1e-2) / scaling), 'S/cm scaled')
            print(key, ': {:.2f}'.format(np.max(device.gms[key].values * 1000)), 'mS max')

    if plot:
        fig = oect_plot.plot_transfers_gm(device, gm_plot=gm_plot, leakage=True)
        fig.savefig(path + r'\transfer_leakage.tif', format='tiff')
        fig = oect_plot.plot_transfers_gm(device, gm_plot=gm_plot, leakage=False)
        fig.savefig(path + r'\transfer.tif', format='tiff')

        fig = oect_plot.plot_outputs(device, leakage=True)
        fig.savefig(path + r'\output_leakage.tif', format='tiff')
        fig = oect_plot.plot_outputs(device, leakage=False)
        fig.savefig(path + r'\output.tif', format='tiff')

    return device


def file_open(caption='Select folder'):
    '''
    File dialog if path not given in load commands
    :param caption:
    :type caption: str
    
    :return:
    :rtype: str
    '''

    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication([])
    path = QtWidgets.QFileDialog.getExistingDirectory(caption=caption)
    app.closeAllWindows()
    app.exit()

    return str(path)


def average(path='', thickness=40e-9, plot=True):
    '''
    averages data in this particular path (for folders 'avg')
    
    :param path: string path to folder '.../avg'. Note Windows path are of form r'Path_name'
    :type path: str
        
    :param thickness: approximate film thickness. Standard polymers (for Raj) are ~40 nm    
    :type thickness: float
        
    :param plot: Whether to plot or not. Not plotting is very fast!
    :type plot: bool
        
    :returns: tuple (pixels, Id_Vg, Id_Vd, temp_dv.WdL)
        WHERE
        dict pixels: dict of OECT contains the various OECT class devices
        DataFrame Id_Vg contains the averaged Id vs Vg (drain current vs gate voltage, transfer)
        DataFrame Id_Vd contains the averaged Id vs Vg (drain current vs drain voltages, output)
    '''

    if not path:
        path = file_open(caption='Select avg subfolder')
        print('Loading from', path)

    filelist = os.listdir(path)

    # removes all but the folders in pixel_names
    f = filelist[:]
    for k in filelist:
        try:
            sub_num = int(k)
        except:
            print('Ignoring', k)
            f.remove(k)
    filelist = f[:]
    del f

    paths = [os.path.join(path, name) for name in filelist]

    # removes random files instead of the sub-folders
    for p in paths:
        if not os.path.isdir(p):
            paths.remove(p)

    pixels = {}
    # loads all the folders
    for p, f in zip(paths, filelist):
        dv = loadOECT(p, params={'d': thickness}, gm_plot=plot, plot=plot)
        pixels[f] = dv

    # average Id-Vg
    Id_Vg = []
    first_pxl = pixels[list(pixels.keys())[0]]

    for dv in pixels:

        if not any(Id_Vg):

            Id_Vg = pixels[dv].transfers.values

        else:

            Id_Vg += pixels[dv].transfers.values

    Id_Vg /= len(pixels)
    Id_Vg = pd.DataFrame(data=Id_Vg)

    try:
        Id_Vg = Id_Vg.set_index(first_pxl.transfers.index)
    except:
        Id_Vg = Id_Vg.set_index(pixels[list(pixels.keys())[-1]])

    # find gm of the average
    temp_dv = oectp.OECT(paths[0], {'d': thickness})
    _gm_fwd, _gm_bwd = temp_dv._calc_gm(Id_Vg)
    Id_Vg['gm_fwd'] = _gm_fwd
    if not _gm_bwd.empty:
        Id_Vg['gm_bwd'] = _gm_bwd

    Id_Vg = Id_Vg.rename(columns={0: 'Id average'})  # fix a naming bug

    if temp_dv.reverse:
        Id_Vg.reverse = True
        Id_Vg.rev_point = temp_dv.rev_point

    # average Id-Vd at max Vd
    Id_Vd = []

    # finds column corresponding to lowest voltage (most doping), but these are strings
    idx = np.argmin(np.array([float(i) for i in first_pxl.outputs.columns]))
    volt = first_pxl.outputs.columns[idx]

    for dv in pixels:

        if not any(Id_Vd):

            Id_Vd = pixels[dv].outputs[volt].values

        else:

            Id_Vd += pixels[dv].outputs[volt].values

    Id_Vd /= len(pixels)
    Id_Vd = pd.DataFrame(data=Id_Vd)

    try:
        Id_Vd = Id_Vd.set_index(pixels[list(pixels.keys())[0]].outputs[volt].index)
    except:
        Id_Vd = Id_Vd.set_index(pixels[list(pixels.keys())[-1]].outputs[volt].index)

    if plot:
        fig = oect_plot.plot_transfer_avg(Id_Vg, temp_dv.WdL)
        fig.savefig(path + r'\transfer_avg.tif', format='tiff')
        fig = oect_plot.plot_output_avg(Id_Vd)
        fig.savefig(path + r'\output_avg.tif', format='tiff')

    return pixels, Id_Vg, Id_Vd, temp_dv.WdL

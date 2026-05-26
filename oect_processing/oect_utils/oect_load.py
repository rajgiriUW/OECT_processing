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
             spline=False,
             verbose=True,
             thickness=None,
             d=None,
             capacitance=None,
             c_star=None,
             params={},
             options={}):
    '''
    Loads all pixel subfolders, processes each OECT, and computes uC*.

    Parameters
    ----------
    path : str
        Path to the folder containing numbered pixel subfolders (e.g. '01', '02').
    plot : list of bool, optional
        [0] Plot the uC* graph; [1] plot individual pixel transfer/output curves.
    V_low : bool, optional
        Detect erroneous turnover points when devices break down.
    retrace_only : bool, optional
        Use only the retrace sweep for each pixel.
    spline : bool, optional
        Use spline-derived gm values instead of the smoothed derivative.
    verbose : bool, optional
        Print progress to display.
    thickness : float, optional
        Film thickness (m). Alias for d; kept for backwards compatibility.
    d : float, optional
        Film thickness (m).
    capacitance : float, optional
        Total capacitance (F). Used as a sanity check against uC*.
    c_star : float, optional
        Volumetric capacitance (F/cm^3) from EIS.
    params : dict, optional
        Additional device parameters passed to each OECT.
    options : dict, optional
        Additional processing options passed to each OECT.

    Returns
    -------
    pixels : dict
        OECT objects keyed by pixel subfolder name.
    uC_dv : dict
        Aggregated results: WdL, W, Vg_Vt, Vt, uC, uC_0, gms, folder, mobility.
    '''
    if not path:
        path = file_open(caption='Select uC subfolder')
        print('Loading from', path)

    # Process files, ignore non-numeric folders (e.g. '01_bad')
    filelist = os.listdir(path)
    f = filelist[:]
    for k in filelist:
        try:
            _ = int(k)
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
    
    # Options is mostly for future-proofing
    opts = {'V_low': V_low}
    if any(options):
        for o in options:
            opts[o] = options[o]
    if thickness: #backwards compatibility
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
            if spline:
                Vg_Vt = np.append(Vg_Vt, pixels[pixel].VgVts_spl)
                gms = np.append(gms, pixels[pixel].gm_peaks_spl['peak gm spline (S)'].values)
            
            else:    
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

    # * 1e-2 to get into right mobility units (cm)
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
    Loads and processes a single OECT pixel folder.

    Parameters
    ----------
    path : str
        Path to the pixel data folder.
    params : dict, optional
        Device parameters, e.g. {'W': 100, 'L': 20, 'd': 40e-9}.
    gm_plot : bool or list of bool, optional
        Passed to plot_transfers_gm as the gm_plot argument.
    plot : bool, optional
        If True, generates and saves transfer and output figures.
    options : dict, optional
        Processing options passed to OECT.
    verbose : bool, optional
        If True, prints peak gm values to display.

    Returns
    -------
    OECT
        Processed pixel object with gms and threshold computed.
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
    Opens a PyQt5 folder dialog and returns the selected path.

    Parameters
    ----------
    caption : str, optional
        Dialog window title.

    Returns
    -------
    str
        Selected folder path.
    '''

    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication([])
    path = QtWidgets.QFileDialog.getExistingDirectory(caption=caption)
    app.closeAllWindows()
    app.exit()

    return str(path)


def average(path='', thickness=40e-9, plot=True):
    '''
    Averages transfer and output curves across pixels in an 'avg' folder.

    Parameters
    ----------
    path : str
        Path to the folder containing numbered pixel subfolders.
    thickness : float, optional
        Film thickness (m). Default is 40 nm.
    plot : bool, optional
        If True, generates and saves averaged transfer and output figures.

    Returns
    -------
    pixels : dict
        OECT objects for each pixel subfolder.
    Id_Vg : DataFrame
        Averaged Id-Vg transfer curve, includes gm_fwd and gm_bwd columns.
    Id_Vd : DataFrame
        Averaged Id-Vd output curve at the most negative Vd.
    WdL : float
        W*d/L prefactor from the first pixel.
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

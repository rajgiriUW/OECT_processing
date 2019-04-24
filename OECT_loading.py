# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:04:47 2018

@author: Raj
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit as cf

import OECT
import OECT_plotting

'''
load_uC : for loading the four pixels to generate a uC* plot
load_avg : for loading all the subfolders that are the 4 "averaging" pixels
    this is somewhat uncommon

Usage:
    
    >> pixels, uC_dv = OECT_loading.uC_scale(r'path_to_uC_scale', new_geom=False)
    
'''


def uC_scale(path='', thickness=40e-9, plot=[True, False], V_low=False, 
             retrace_only=False, verbose=True, options={}):
    '''
    path: str
        string path to folder '.../avg'. Note Windows path are of form r'Path_name'
      
    thickness : float
        approximate film thickness. Standard polymers (for Raj) are ~40 nm
        
    plot : list of bools, optional
        [0]: Plot the uC* data
        [1]: plot the individual plots
        Whether to plot or not. Not plotting is very fast!    

    retrace_only : bool, optional
        Whether to only do the retrace in case trace isn't saturating
        
    V_low : bool, optional
        Whether to find erroneous "turnover" points when devices break down

    verbose: bool, optional
        Print to display

    Returns
    -------
    pixels : dict of OECT
        Contains the various OECT class devices
        
    uC_dv : dict containing
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
    

    # loads all the folders
    if type(plot)==bool or len(plot) == 1:
        plot = [plot, plot]
        
    for p, f in zip(paths, pixkeys):

        if os.listdir(p):

            if verbose:
                print(p)
            dv = loadOECT(p, {'d': thickness}, gm_plot=plot, plot=plot[1],
                          options=opts, verbose=verbose)
            pixels[f] = dv

        else:

            pixkeys.remove(f)

    # do uC* graphs, need gm vs W*d/L
    Wd_L = np.array([])
    W = np.array([])
    Vg_Vt = np.array([])  # threshold offset
    Vt = np.array([])
    gms = np.array([])

    # assumes Length and thickness are fixed
    uC_dv = {}

    for pixel in pixels:
        
        if not pixels[pixel].gms.empty:
            
            ix = len(pixels[pixel].VgVts)
            Vt = np.append(Vt, pixels[pixel].Vts)
            Vg_Vt = np.append(Vg_Vt, pixels[pixel].VgVts)
            gms = np.append(gms, pixels[pixel].gm_peaks['peak gm (S)'].values)
            W = np.append(W, pixels[pixel].W)
            
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

    if plot[0]:
        fig = OECT_plotting.plot_uC(uC_dv)

        if verbose:
            print('uC* = ', str(uC_0 * 1e-2), ' F/cm*V*s')

    if verbose: 
        print('Vt = ', uC_dv['Vt'])

    return pixels, uC_dv

def loadOECT(path, params=None, gm_plot=True, plot=True, options={}, verbose=True):
    """
    Wrapper function for processing OECT data

    params = {W: , L: , d: } for W, L, d of device

    USAGE:
        device1 = loadOECT(folder_name)
    """

    if not path:
        path = file_open(caption='Select device subfolder')

    device = OECT.OECT(path, params, options)
    device.calc_gms()
    device.thresh()

    scaling = device.WdL  # W *d / L

    if verbose:
        for key in device.gms:
            print(key, ': {:.2f}'.format(np.max(device.gms[key].values * 1e-2) / scaling), 'S/cm scaled')
            print(key, ': {:.2f}'.format(np.max(device.gms[key].values*1000)), 'mS max')

    if plot:
        fig = OECT_plotting.plot_transfers_gm(device, gm_plot=gm_plot, leakage=True)
        fig.savefig(path + r'\transfer_leakage.tif', format='tiff')
        fig = OECT_plotting.plot_transfers_gm(device, gm_plot=gm_plot, leakage=False)
        fig.savefig(path + r'\transfer.tif', format='tiff')

        fig = OECT_plotting.plot_outputs(device, leakage=True)
        fig.savefig(path + r'\output_leakage.tif', format='tiff')
        fig = OECT_plotting.plot_outputs(device, leakage=False)
        fig.savefig(path + r'\output.tif', format='tiff')

    return device


def file_open(caption='Select folder'):
    '''
    File dialog if path not given in load commands
    :param
        caption : str

    :return:
        path : str
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

    path: str
        string path to folder '.../avg'. Note Windows path are of form r'Path_name'

    thickness : float
        approximate film thickness. Standard polymers (for Raj) are ~40 nm

    plot : bool
        Whether to plot or not. Not plotting is very fast!


    Returns
    -------
    pixels : dict of OECT
        Contains the various OECT class devices

    Id_Vg : pandas dataframe
        Contains the averaged Id vs Vg (drain current vs gate voltage, transfer)

    Id_Vd : pandas dataframe
        Contains the averaged Id vs Vg (drain current vs drain voltages, output)

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
    temp_dv = OECT.OECT(paths[0], {'d': thickness})
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
        fig = OECT_plotting.plot_transfer_avg(Id_Vg, temp_dv.WdL)
        fig.savefig(path + r'\transfer_avg.tif', format='tiff')
        fig = OECT_plotting.plot_output_avg(Id_Vd)
        fig.savefig(path + r'\output_avg.tif', format='tiff')

    return pixels, Id_Vg, Id_Vd, temp_dv.WdL
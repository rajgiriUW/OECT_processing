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
load_avg : for loading all the subfolders that are the 4 "averaging" pixels
load_uC : for loading the four pixels to generate a uC* plot

Usage:
    
    >> pixels, Id_Vg, Id_Vd = OECT_loading.average(r'path_to_avg', new_geom=False)
    >> pixels, uC_dv = OECT_loading.uC_scale(r'path_to_uC_scale', new_geom=False)
    
'''


def uC_scale(path='', thickness=40e-9, plot=[True, False], add_avg_pixels=True, V_low=False):
    '''
    path: str
        string path to folder '.../avg'. Note Windows path are of form r'Path_name'
      
    thickness : float
        approximate film thickness. Standard polymers (for Raj) are ~40 nm
        
    plot : list of bools, optional
        [0]: Plot the uC* data
        [1]: plot the individual plots
        Whether to plot or not. Not plotting is very fast!    

    add_avg_pixels : bool, optional
        Whether to add the averaging subfolder (adds more low Wd/L points)
        
    V_low : bool, optional
        Whether to find erroneous "turnover" points when devices break down

    Returns
    -------
    pixels : dict of OECT
        Contains the various OECT class devices
        
    uC_dv : OECT Class containing
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
            print('Ignoring', k)
            f.remove(k)
    filelist = f[:]
    paths = [os.path.join(path, name) for name in filelist]
    pixkeys = [f + '_uC' for f in filelist]

    # add the averaging pixels to the calculation
    if add_avg_pixels:

        try:

            os.chdir(path + '\..')
            os.chdir(os.getcwd() + '\\avg')
            avgpath = os.getcwd()
            avglist = os.listdir(avgpath)

            print('Adding avg-pixels')

            f = avglist[:]
            for k in avglist:
                try:
                    sub_num = int(k)
                except:
                    print('Ignoring', k)
                    f.remove(k)

            filelist = f[:]
            paths = paths + [os.path.join(avgpath, name) for name in filelist]
            pixkeys = pixkeys + [f + '_avg' for f in filelist]
            del f

        except:

            print('No avg subfolder found')

    # removes random files instead of the sub-folders
    for p in paths:
        if not os.path.isdir(p):
            paths.remove(p)

    pixels = {}

    # loads all the folders
    for p, f in zip(paths, pixkeys):

        if os.listdir(p):

            print(p)
            dv = loadOECT(p, {'d': thickness}, gm_plot=plot, plot=plot[1],
                          options={'V_low': V_low})
            pixels[f] = dv

        else:

            pixkeys.remove(f)

    # do uC* graphs, need gm vs W*d/L
    Wd_L = np.array([])
    Vg_Vt = np.array([])  # threshold offset
    Vt = np.array([])
    gms = np.array([])

    for f, pixel in zip(pixkeys, pixels):

        # peak gms
        c = list(pixels[pixel].gm_fwd.keys())[0]

        if not pixels[pixel].gm_fwd[c].empty:
            gm_fwd = np.max(pixels[pixel].gm_fwd[c].values)
            Wd_L = np.append(Wd_L, pixels[pixel].WdL)
            Vt = np.append(Vt, pixels[pixel].Vts[0])
            Vg_Vt = np.append(Vg_Vt, pixels[pixel].VgVts[0])
            gms = np.append(gms, gm_fwd)

        # backwards
        c = list(pixels[pixel].gm_bwd.keys())[0]

        if not pixels[pixel].gm_bwd[c].empty:
            gm_bwd = np.max(pixels[pixel].gm_bwd[c].values)
            Wd_L = np.append(Wd_L, pixels[pixel].WdL)
            Vt = np.append(Vt, pixels[pixel].Vts[1])
            Vg_Vt = np.append(Vg_Vt, pixels[pixel].VgVts[1])
            gms = np.append(gms, gm_bwd)

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
    uC_dv = {'W': 100e-6, 'L': 20e-6, 'd': thickness,
             'WdL': Wd_L,
             'Vg_Vt': Vg_Vt,
             'Vt': Vt,
             'uC': uC,
             'uC_0': uC_0,
             'gms': gms,
             'folder': path}

    if plot[0]:
        fig = OECT_plotting.plot_uC(uC_dv)

        print('uC* = ', str(uC_0 * 1e-2), ' F/cm*V*s')

    print('Vt = ', uC_dv['Vt'])

    return pixels, uC_dv


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


def loadOECT(path, params, gm_plot=True, plot=True, options={}):
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

    for key in device.gms:
        print(key, ':', np.max(device.gms[key].values) / scaling, 'S/m scaled')
        print(key, ':', np.max(device.gms[key].values), 'S max')

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

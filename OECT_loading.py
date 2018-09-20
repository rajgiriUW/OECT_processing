# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:04:47 2018

@author: Raj
"""

import pandas as pd
import os

import numpy as np
from matplotlib import pyplot as plt

import OECT_plotting 
import OECT

from scipy.optimize import curve_fit as cf

'''
Load_avg : for loading all the subfolders that are the 4 "averaging" pixels
load_uC : for loading the four pixels to generate a uC* plot

'''

def load_avg(path, thickness = 40e-9, plot=True):
    '''
    averages data in this particular path (for folders 'avg')
    
    path should be to folder '.../avg' 
    
    thickness  of the film
    
    Returns
    -------
    pixels : dict of OECT
        Contains the various OECT class devices
        
    Id_Vg : pandas dataframe
        Contains the averaged Id vs Vg (drain current vs gate voltage, transfer)
        
    Id_Vd : pandas dataframe
        Contains the averaged Id vs Vg (drain current vs drain voltages, output)
        
    '''

    params = {'W': 100e-6, 'L': 100e-6, 'd': thickness}
    
    filelist = os.listdir(path)
    pixel_names = ['01', '02', '03', '04']
    
    # removes all but the folders in pixel_names
    f = filelist[:]
    for k in filelist:
        if k not in pixel_names:
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
        
        dv = loadOECT(p, params, gm_plot=plot, plot=plot)
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
    Id_Vg = pd.DataFrame(data= Id_Vg)
    
    try:
        Id_Vg = Id_Vg.set_index(first_pxl.transfers.index)
    except: 
        Id_Vg = Id_Vg.set_index(pixels[list(pixels.keys())[-1]])
    
    # find gm of the average
    temp_dv = OECT.OECT(path, params)
    _gm_fwd, _gm_bwd = temp_dv._calc_gm(Id_Vg)
    Id_Vg['gm_fwd'] = _gm_fwd
    Id_Vg['gm_bwd'] = _gm_bwd
    Id_Vg = Id_Vg.rename(columns = {0: 'Id average'}) # fix a naming bug
    del temp_dv
    
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
    Id_Vd = pd.DataFrame(data= Id_Vd)
    
    try:
        Id_Vd = Id_Vd.set_index(pixels[list(pixels.keys())[0]].outputs[volt].index)
    except:
        Id_Vd = Id_Vd.set_index(pixels[list(pixels.keys())[-1]].outputs[volt].index)
    
    if plot:
            fig = OECT_plotting.plot_transfer_avg(Id_Vg)
            fig.savefig(path+r'\transfer_avg.tif', format='tiff')
            fig = OECT_plotting.plot_output_avg(Id_Vd)
            fig.savefig(path+r'\output_avg.tif', format='tiff')
    
    return pixels, Id_Vg, Id_Vd


def uC_scale(path, thickness=40e-9, plot=True):
    '''
    01 = 2000/20
    02 = 1000/20
    03 = 200/20
    04 = 50/20
    05 = 100/100 (using an "averaging" pixel as a fifth data point)
    
    From Lucas:
        100umx100 um on the top 4 and 2000/20 1000/20 200/20 50/20 for the bottom row
    
    Returns
    -------
    pixels : dict of OECT
        Contains the various OECT class devices
        
    Wd_L : ndarray
        coefficient for plotting on x-axis
    
    gms : ndarray
        average transconductance for plotting on y-axis
    
    Vg_Vt : ndarray
        threshold voltage shifts for correcting uC* fit
    
    '''
    
    filelist = os.listdir(path)
    pixel_names = ['01', '02', '03', '04', '05']
    
    f = filelist[:]
    for k in filelist:
        if k not in pixel_names:
            f.remove(k)
    filelist = f[:]
    del f

    params_super = {'01': {'W': 2000e-6 ,'L': 20e-6, 'd': thickness},
                    '02': {'W': 1000e-6, 'L': 20e-6, 'd': thickness},
                    '03': {'W': 200e-6, 'L': 20e-6, 'd': thickness},
                    '04': {'W': 50e-6, 'L': 20e-6, 'd': thickness},
                    '05': {'W': 100e-6, 'L': 100e-6, 'd': thickness}
                    }

    paths = [os.path.join(path, name) for name in filelist]

    # removes random files instead of the sub-folders
    for p in paths:
        if not os.path.isdir(p):
            paths.remove(p)
    
    pixels = {}
    # loads all the folders
    for p, f in zip(paths, filelist):
        
        dv = loadOECT(p, params_super[f], gm_plot=plot, plot=plot)
        pixels[f] = dv

    # do uC* graphs, need gm vs W*d/L        
    Wd_L = np.array([])
    Vg_Vt = np.array([]) # threshold offset
    Vt = np.array([])
    gms = np.array([])
    
    for f, pixel in zip(filelist, pixels):
        Wd_L = np.append(Wd_L, params_super[f]['W']*thickness/params_super[f]['L'])
        
        # peak gms
        reverse = False
        c = list(pixels[pixel].gm_fwd.keys())[0]
        
        if not pixels[pixel].gm_fwd[c].empty:
            
            gm_fwd = np.max(pixels[pixel].gm_fwd[c].values)
            gm_argmax = np.argmax(pixels[pixel].gm_fwd[c].values)
            
            Vg_fwd = pixels[pixel].gm_fwd[c].index[gm_argmax]
            Vg_Vt_fwd = pixels[pixel].Vts[0] - Vg_fwd
        
        # backwards
        c = list(pixels[pixel].gm_bwd.keys())[0]
        
        if not pixels[pixel].gm_bwd[c].empty:
            reverse = True
            gm_bwd = np.max(pixels[pixel].gm_bwd[c].values)
        
            gm_argmax = np.argmax(pixels[pixel].gm_bwd[c].values)
            Vg_bwd = pixels[pixel].gm_bwd[c].index[gm_argmax]
            Vt = np.append(Vt, pixels[pixel].Vts[1])
            Vg_Vt_bwd = pixels[pixel].Vts[1] - Vg_bwd
        
        if reverse:
            gm = np.mean([gm_fwd, gm_bwd])
            gms = np.append(gms, gm)
            
            Vg_Vt = np.append(Vg_Vt, np.mean([Vg_Vt_fwd, Vg_Vt_bwd]))
            
        else:
            gms = np.append(gms, gm_fwd)
            Vg_Vt = np.append(Vg_Vt, pixels[pixel].Vts[0] - Vg_fwd)
        
        Vt = np.append(Vt, pixels[pixel].Vt)
            
    # fit functions
    def line_f(x, a, b):
        
        return a + b*x
    
    def line_0(x, b):
        'no y-offset --> better log-log fits'
        return b * x

    # * 1e2 to get into right mobility units (cm)
    uC_0, _ = cf(line_0, Wd_L*Vg_Vt, gms)
    uC, _ = cf(line_f, Wd_L*Vg_Vt, gms)

    if plot:
        
        fig, ax = plt.subplots(facecolor='white', figsize=(10,8))
        ax.plot(np.abs(Wd_L*Vg_Vt)*1e2, gms*1000, 's', markersize=10, color='b')
        ax.set_xlabel('Wd/L * (Vg-Vt) (cm*V)')
        ax.set_ylabel('gm (mS)')
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.set_title('uC* = ' + str(uC_0*1e-2)+' F/cm*V*s')
        fig.savefig(path+r'\scaling_uC.tif', format='tiff')
        
        Wd_L_fitx = np.arange(Wd_L[-1]*Vg_Vt[-1], Wd_L[0]*Vg_Vt[0], 1e-9)
        ax.plot(Wd_L_fitx*1e2, (uC[1]*Wd_L_fitx + uC[0])*1000, 'k--')
        ax.plot(Wd_L_fitx*1e2, (uC_0[0]*Wd_L_fitx)*1000, 'r--')
        ax.set_title('uC* = ' + str(uC_0*1e-2)+' F/cm*V*s')
        fig.savefig(path+r'\scaling_uC_+fit.tif', format='tiff')
        
        fig, ax = plt.subplots(facecolor='white', figsize=(10,8))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(np.abs(Wd_L*Vg_Vt)*1e2, gms, 's', markersize=6)
        ax.set_xlabel('Wd/L * (Vg-Vt) (cm*V)')
        ax.set_ylabel('gm (S)')
        ax.plot(Wd_L_fitx*1e2, (uC[1]*Wd_L_fitx + uC[0]), 'k--')
        ax.plot(Wd_L_fitx*1e2, (uC_0[0]*Wd_L_fitx), 'r--')
        ax.set_title('uC* = ' + str(uC_0*1e-2)+' F/cm*V*s')
        fig.savefig(path+r'\scaling_uC_loglog.tif', format='tiff')
        
        print('uC* = ',str(uC_0*1e-2),' F/cm*V*s')
        
    return pixels, Wd_L, gms, Vg_Vt, uC

def loadOECT(path, params, gm_plot=True, plot=True):
    """
    Wrapper function for processing OECT data

    params = {W: , L: , d: } for W, L, d of device

    USAGE:
        device1 = loadOECT(folder_name)
        

    """

    device = OECT.OECT(path, params)
    device.loaddata()
    device.calc_gms()
    device.thresh()
    
    scaling = params['W'] * params['d']/params['L']
    
    for key in device.gms_fwd:
        print(key,':', np.max(device.gms_fwd[key].values)/scaling, 'S/m scaled'  )
        print(key,':', np.max(device.gms_fwd[key].values), 'S max' )

    if plot:
    
        fig = OECT_plotting.plot_transfers_gm(device, gm_plot=gm_plot, leakage=True)
        fig.savefig(path+r'\transfer_leakage.tif', format='tiff')
        fig = OECT_plotting.plot_transfers_gm(device, gm_plot=gm_plot, leakage=False)
        fig.savefig(path+r'\transfer.tif', format='tiff')   
    
        fig = OECT_plotting.plot_outputs(device, leakage=True)
        fig.savefig(path+r'\output_leakage.tif', format='tiff')
        fig = OECT_plotting.plot_outputs(device, leakage=False)
        fig.savefig(path+r'\output.tif', format='tiff')

    return device
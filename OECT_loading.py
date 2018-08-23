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

'''
Load_avg : for loading all the subfolders that are the 4 "averaging" pixels
load_uC : for loading the four pixels to generate a uC* plot

'''

def load_avg(path, thickness = 30e-9, plot=True):
    '''
    averages data in this particular path (for folders 'avg')
    
    path should be to folder '.../avg' 
    
    thickness  of the film
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
        
        dv = loadOECT(p, params, gm_plot=plot)
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
    Id_Vg['gm'] = temp_dv.calc_gm(Id_Vg)[0]
    Id_Vg = Id_Vg.rename(columns = {0: 'avg'}) # fix a naming bug
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


def uC_scale(path, thickness=30e-9, plot=True):
    '''
    01 = 2000/20
    02 = 1000/20
    03 = 200/20
    04 = 50/20
    05 = 100/100 (using an "averaging" pixel as a fifth data point)
    
    From Lucas:
        100umx100 um on the top 4 and 2000/20 1000/20 200/20 50/20 for the bottom row
    
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
        
        dv = loadOECT(p, params_super[f], gm_plot=plot)
        pixels[f] = dv

    # do uC* graphs, need gm vs W*d/L        
    Wd_L = np.array([])
    gms = np.array([])
    
    for f, pixel in zip(filelist, pixels):
        Wd_L = np.append(Wd_L, params_super[f]['W']*thickness/params_super[f]['L'])
        
        c = list(pixels[pixel].gms_fwd.keys())[0]
        gm = np.max(pixels[pixel].gms_fwd[c].values)
        gms = np.append(gms, gm)
        
    if plot:
        
        fig, ax = plt.subplots(facecolor='white', figsize=(10,8))
        ax.plot(Wd_L*1e9, gms*1000, 's')
        ax.set_xlabel('Wd/L (nm)')
        ax.set_ylabel('gm (mS)')
        fig.savefig(path+r'\scaling_uC.tif', format='tiff')
        
    return pixels, Wd_L, gms

def loadOECT(path, params, gm_plot=True):
    """
    Wrapper function for processing OECT data

    params = {W: , L: , d: } for W, L, d of device

    USAGE:
        device1 = loadOECT(folder_name)
        

    """

    device = OECT.OECT(path, params)
    device.loaddata()
    device.calc_gms()
    
    scaling = params['W'] * params['d']/params['L']
    
    for key in device.gms_fwd:
        print(key,':', np.max(device.gms_fwd[key].values)/scaling, 'S/m scaled'  )
        print(key,':', np.max(device.gms_fwd[key].values), 'S max' )

    fig = OECT_plotting.plot_transfers_gm(device, gm_plot=gm_plot, leakage=True)
    fig.savefig(path+r'\transfer_leakage.tif', format='tiff')
    fig = OECT_plotting.plot_transfers_gm(device, gm_plot=gm_plot, leakage=False)
    fig.savefig(path+r'\transfer.tif', format='tiff')   

    fig = OECT_plotting.plot_outputs(device, leakage=True)
    fig.savefig(path+r'\output_leakage.tif', format='tiff')
    fig = OECT_plotting.plot_outputs(device, leakage=False)
    fig.savefig(path+r'\output.tif', format='tiff')

    return device
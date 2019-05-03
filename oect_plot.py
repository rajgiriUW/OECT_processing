# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:17:40 2017

@author: Raj
"""

import warnings
from itertools import cycle

import matplotlib.cbook
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

'''
### PLOTTING FUNCTIONS ###

2 primary plotting functions here of use:
    -plot_uC : 
        Generates a uC* graph
        Usually passed by OECT_loading.uC_scale
        Requires a dictionary of points
    -plot_transfers_gm : 
        Generates a Id-Vg and gm-Vg graph
        Requires an OECT device object (e.g. dv = OECT.OECT(...), dv is the object)
    -plot_outputs :
        Generates a Id-Vd graph for all the output curves in a passed device
        Requires an OECT device object

The other plotting functions are mostly used in OECT_loading or in OECT itself

'''

def plot_uC(dv, label='', savefig=True, ax=None, fit=True,
            **kwargs):
    """
    dv : dict of parameters needed for plotting

        This dict needs the following:
            WdL : array
                W * d /L values for the pixels
            Vg_Vt : array
                Vg-Vt (gate - threshold) for the pixels
            gms : array
                transconductances for the pixels
            path : string
                for saving the resulting graph
            uC : 2-element array 
                For generating a y = mx + b line
            uC_0 : 1-element array 
                For generating a y = mx line

    label : str, optional
        For saving files, incldues this in the filename

    savefig : bool, optional
        Whether to save the figures
        
    ax : matplotlib axes object, optional
        Plot on an existing axis
        
       
    fit : bool, optional
        Plot the best fit line or not
        
    kwargs : dict, optional
        Standard plotting params, e.g. {'color': 'b'}
        Default is size 10 blue squares 
    """
    WdL = dv['WdL']
    Vg_Vt = dv['Vg_Vt']
    uC = dv['uC']
    uC_0 = dv['uC_0']
    gms = dv['gms']
    
    if savefig:
        if 'folder' in dv:
            path = dv['folder']
        else:
            import os
            path = os.getcwd()
    
    if not np.any(ax):
        fig, ax = plt.subplots(nrows=2, facecolor='white', figsize=(9, 12))

    params = {'color': 'b', 'markersize': 10, 'marker': 's', 'linestyle': ''}
    
    for k in kwargs:
        params[k] = kwargs[k]
    
    plt.rcParams.update({'font.size': 24, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})
    plt.rc('axes', linewidth=4)
    ax[0].tick_params(labeltop=False, labelright=False)
    ax[0].tick_params(axis='both', length=14, width=3, which='major',
                   bottom='on', left='on', right='on', top='on')
    ax[0].tick_params(axis='both', length=10, width=3, which='minor',
                   bottom='on', left='on', right='on', top='on')

    ax[0].plot(np.abs(WdL * Vg_Vt) * 1e2, gms * 1000, **params)
    ax[0].set_xlabel('Wd/L * (Vg-Vt) (cm*V)')
    ax[0].set_ylabel('gm (mS)')
    ax[0].xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax[0].set_title('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
    plt.tight_layout()

    if savefig:
        fig = plt.gcf()
        fig.savefig(path + r'\scaling_uC' + label + '.tif', format='tiff')

    if fit:
        # create x-axis for fits
        _xl = np.argmin(WdL * Vg_Vt)
        _xh = np.argmax(WdL * Vg_Vt)
        Wd_L_fitx = np.arange(WdL[_xl] * Vg_Vt[_xl], WdL[_xh] * Vg_Vt[_xh], 1e-9)
        #ax.plot(Wd_L_fitx * 1e2, (uC[1] * Wd_L_fitx + uC[0]) * 1000, 'k--')
        ax[0].plot(Wd_L_fitx * 1e2, (uC_0[0] * Wd_L_fitx) * 1000, 'r--')
        ax[0].set_title('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
        ax[0].tick_params(axis='both', length=17, width=3, which='major',
                       bottom='on', left='on', right='on', top='on')
        ax[0].tick_params(axis='both', length=10, width=3, which='minor',
                       bottom='on', left='on', right='on', top='on')
        plt.tight_layout()
    
        print('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
    
        if savefig:
            fig.savefig(path + r'\scaling_uC_+fit' + label + '.tif', format='tiff')

    #fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].plot(np.abs(WdL * Vg_Vt) * 1e2, gms, **params)
    ax[1].set_xlabel('Wd/L * (Vg-Vt) (cm*V)')
    ax[1].set_ylabel('gm (S)')
    #ax.plot(Wd_L_fitx * 1e2, (uC[1] * Wd_L_fitx + uC[0]), 'k--')
    if fit:
        ax[1].plot(Wd_L_fitx * 1e2, (uC_0[0] * Wd_L_fitx), 'r--')
        
    ax[1].set_title('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
    ax[1].tick_params(axis='both', length=17, width=3, which='major',
                   bottom='on', left='on', right='on', top='on')
    ax[1].tick_params(axis='both', length=10, width=3, which='minor',
                   bottom='on', left='on', right='on', top='on')
    plt.tight_layout()

    if savefig:
        fig.savefig(path + r'\scaling_uC_loglog' + label + '.tif', format='tiff')

    return ax

def plot_transfers_gm(dv, gm_plot=True, leakage=False):
    ''' 
    For plotting transfer and gm on the same plot for one pixel
            
    '''

    fig, ax1 = plt.subplots(facecolor='white', figsize=(10, 8))
    ax2 = ax1.twinx()

    plt.rc('axes', linewidth=4)
    plt.rcParams.update({'font.size': 18, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})

    ax1.tick_params(axis='both', length=10, width=3, which='major', top='on')
    ax1.tick_params(axis='both', length=6, width=3, which='minor', top='on')
    ax1.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='both', length=10, width=3, which='major')
    ax2.tick_params(axis='both', length=6, width=3, which='minor')
    ax1.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    markers = ['o', 's', '^', 'd', 'x']
    mk = cycle(markers)

    for k in dv.transfers.columns:

        if dv.reverse:
            ax1.plot(dv.transfers[k][:dv.rev_point] * 1000,
                     linewidth=2, marker=next(mk), markersize=7, color='b')
            ax1.plot(dv.transfers[k][dv.rev_point:] * 1000,
                     linewidth=2, marker=next(mk), markersize=7, color='r')

        else:
            ax1.plot(dv.transfers[k][:] * 1000,
                     linewidth=2, marker=next(mk), markersize=7, color='b')

    if leakage:
        for k in dv.transfer_raw:
           ax1.plot(dv.transfer_raw[k].index, dv.transfer_raw[k]['I_G (A)'] * 1000,
                     linewidth=1, linestyle='--')

    markers = ['o', 's', '^', 'd', 'x']
    mk = cycle(markers)

    if gm_plot:
        for k in dv.gms:
            ax2.plot(dv.gms[k].index, dv.gms[k] * 1000, 'b--', linewidth=2)

    ax1.set_ylabel('Ids Current (mA)', fontweight='bold',
                   fontsize=18, fontname='Arial')
    ax2.set_ylabel('Transconductance (mS)', rotation=-90, labelpad=20,
                   fontweight='bold', fontname='Arial', fontsize=18)
    ax1.set_xlabel('Vgs Voltage (V)', fontweight='bold', fontname='Arial',
                   fontsize=18)

    xminor = AutoMinorLocator(4)
    ax1.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax1.yaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax2.yaxis.set_minor_locator(xminor)

    plt.title(dv.folder, y=1.05)

    return fig

def plot_outputs(dv, leakage=False):
    '''
    dv : OECT class object
    '''

    fig, ax = plt.subplots(facecolor='white', figsize=(12, 8))

    if leakage:
        ax2 = ax.twinx()

    plt.rc('axes', linewidth=4)
    ax.tick_params(labeltop=False, labelright=False)
    ax.tick_params(axis='both', length=10, width=3, which='major',
                   bottom='on', left='on', right='on', top='on', labelsize=18)
    ax.tick_params(axis='both', length=6, width=3, which='minor',
                   bottom='on', left='on', right='on', top='on', labelsize=18)
    plt.rcParams.update({'font.size': 24, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})

    markers = ['o', 's', '^', 'd', 'x']
    mk = cycle(markers)

    for k in dv.outputs.columns:
        ax.plot(dv.outputs.index, dv.outputs[k] * 1000,
                linewidth=2, marker=next(mk), markersize=7)

        if leakage:
            ax2.plot(dv.outputs.index, dv.output_raw[k]['I_G (A)'] * 1000,
                     linewidth=1, linestyle='--')

    ax.legend(labels=dv.Vg_labels, frameon=False,
              fontsize=16, loc=4)
    ax.set_ylabel('Ids Current (mA)', fontweight='bold', fontname='Arial', fontsize=18)
    ax.set_xlabel('Vds Voltage (V)', fontweight='bold', fontname='Arial', fontsize=18)

    if leakage:
        ax2.set_ylabel('Gate leakage (mA)', fontweight='bold', fontname='Arial',
                       fontsize=18, rotation=-90)

    xminor = AutoMinorLocator(4)
    ax.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(xminor)

    plt.title(dv.folder, y=1.05)

    return fig

def plot_output_avg(dv):
    '''
    called by OECT_loading functions
    
    dv = dataFrame
    '''

    fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))

    plt.rc('axes', linewidth=4)
    ax.tick_params(labeltop=False, labelright=False)
    ax.tick_params(axis='both', length=10, width=3, which='major',
                   bottom='on', left='on', right='on', top='on')
    ax.tick_params(axis='both', length=6, width=3, which='minor',
                   bottom='on', left='on', right='on', top='on')
    plt.rcParams.update({'font.size': 24, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})

    ax.plot(dv * 1000, marker='o')

    ax.set_ylabel('Ids Current (mA)', fontweight='bold', fontname='Arial')
    ax.set_xlabel('Vds Voltage (V)', fontweight='bold', fontname='Arial')

    xminor = AutoMinorLocator(4)
    ax.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(xminor)

    plt.title('Output Average', y=1.05)

    return fig

def plot_transfer_avg(dv, Wd_L, label=''):
    ''' 
    For plotting averaged data
    
    This plots transfer and outputs for multiple datasets on the same pixel
    
    '''
    fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))
    ax2 = ax.twinx()

    plt.rc('axes', linewidth=4)
    ax.tick_params(labeltop=False, labelright=False)
    ax.tick_params(axis='both', length=10, width=3, which='major',
                   bottom='on', left='on', right='on', top='on')
    ax.tick_params(axis='both', length=6, width=3, which='minor',
                   bottom='on', left='on', right='on', top='on')
    plt.rcParams.update({'font.size': 24, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})

    if dv.reverse:
        ax.plot(dv['Id average'][:dv.rev_point] * 1000, marker='o', color='b')
        ax.plot(dv['Id average'][dv.rev_point:] * 1000, marker='o', color='r')
    else:
        ax.plot(dv['Id average'] * 1000, marker='o', color='b')

    ax2.plot(dv['gm_fwd'] * 1000, linestyle='--', color='b')
    ax2.plot(dv['gm_bwd'] * 1000, linestyle='--', color='r')
    ax2.set_ylabel('Transconductance (mS)', rotation=-90, labelpad=20,
                   fontweight='bold', fontname='Arial', fontsize=18)

    ax.set_ylabel('Ids Current (mA)', fontweight='bold', fontname='Arial')
    ax.set_xlabel('Vgs Voltage (V)', fontweight='bold', fontname='Arial')

    xminor = AutoMinorLocator(4)
    ax.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(xminor)

    plt.tight_layout()
    plt.title('Average', y=1.05)

    # normalized gm
    fig, ax = plt.subplots(facecolor='white', figsize=(12, 9))
    ax2 = ax.twinx()

    plt.rc('axes', linewidth=4)
    ax.tick_params(labeltop=False, labelright=False)
    ax.tick_params(axis='both', length=10, width=3, which='major',
                   bottom='on', left='on', right='on', top='on')
    ax.tick_params(axis='both', length=6, width=3, which='minor',
                   bottom='on', left='on', right='on', top='on')
    plt.rcParams.update({'font.size': 24, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})

    if dv.reverse:
        ax.plot(dv['Id average'][:dv.rev_point] * 1000, marker='o', color='b')
        ax.plot(dv['Id average'][dv.rev_point:] * 1000, marker='o', color='r')
    else:
        ax.plot(dv['Id average'] * 1000, marker='o', color='b')

    ax2.plot(dv['gm_fwd'] * 1000 / (1e9 * Wd_L), linestyle='--', color='b')
    ax2.plot(dv['gm_bwd'] * 1000 / (1e9 * Wd_L), linestyle='--', color='r')

    ax2.set_ylabel('Norm gm (mS/nm)', rotation=-90, labelpad=20,
                   fontweight='bold', fontname='Arial', fontsize=18)

    ax.set_ylabel('Ids Current (mA)', fontweight='bold', fontname='Arial')
    ax.set_xlabel('Vgs Voltage (V)', fontweight='bold', fontname='Arial')

    xminor = AutoMinorLocator(4)
    ax.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(xminor)
    plt.title('Average', y=1.05)
    plt.tight_layout()

    return fig

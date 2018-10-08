# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:17:40 2017

@author: Raj
"""

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from itertools import cycle

""" PLOTTING FUNCTIONS """
def plot_transfer(dv):

    plt.figure()
    plt.plot(dv.transfer.index, dv.transfer)
    
    
def plot_uC(dv):
    
    Wd_L = dv.Wd_L
    Vg_Vt = dv.Vg_Vt
    uC = dv.uC
    uC_0 = dv.uC_0
    gms = dv.gms
    path = dv.folder    
    
    fig, ax = plt.subplots(facecolor='white', figsize=(10,8))

    plt.rcParams.update({'font.size': 24, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})
    plt.rc('axes', linewidth=4)
    ax.tick_params(labeltop=False, labelright=False)
    ax.tick_params(axis='both', length=14, width=3, which='major',
                    bottom='on', left='on', right='on', top='on')
    ax.tick_params(axis='both', length=10, width=3, which='minor',
                    bottom='on', left='on', right='on', top='on')

    ax.plot(np.abs(Wd_L*Vg_Vt)*1e2, gms*1000, 's', markersize=10, color='b')
    ax.set_xlabel('Wd/L * (Vg-Vt) (cm*V)')
    ax.set_ylabel('gm (mS)')
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.set_title('uC* = ' + str(uC_0*1e-2)+' F/cm*V*s')
    plt.tight_layout()
    fig.savefig(path+r'\scaling_uC.tif', format='tiff')
    
    # create x-axis for fits
    _xl = np.argmin(Wd_L)
    _xh = np.argmax(Wd_L)
    Wd_L_fitx = np.arange(Wd_L[_xl]*Vg_Vt[_xl], Wd_L[_xh]*Vg_Vt[_xh], 1e-9)
    ax.plot(Wd_L_fitx*1e2, (uC[1]*Wd_L_fitx + uC[0])*1000, 'k--')
    ax.plot(Wd_L_fitx*1e2, (uC_0[0]*Wd_L_fitx)*1000, 'r--')
    ax.set_title('uC* = ' + str(uC_0*1e-2)+' F/cm*V*s')
    ax.tick_params(axis='both', length=17, width=3, which='major',
                   bottom='on', left='on', right='on', top='on')
    ax.tick_params(axis='both', length=10, width=3, which='minor',
                    bottom='on', left='on', right='on', top='on')
    plt.tight_layout()
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
    ax.tick_params(axis='both', length=17, width=3, which='major',
                   bottom='on', left='on', right='on', top='on')
    ax.tick_params(axis='both', length=10, width=3, which='minor',
                    bottom='on', left='on', right='on', top='on')
    plt.tight_layout()
    fig.savefig(path+r'\scaling_uC_loglog.tif', format='tiff')
    
    return fig

def plot_transfer_avg(dv, params):
    
    fig, ax = plt.subplots(facecolor='white', figsize=(10,8))
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
        ax.plot(dv['Id average'][:dv.rev_point]*1000, marker='o', color='b')
        ax.plot(dv['Id average'][dv.rev_point:]*1000, marker='o', color='r')
    else:
        ax.plot(dv['Id average']*1000, marker='o', color='b')
        
    ax2.plot(dv['gm_fwd']*1000, linestyle='--', color='b')
    ax2.plot(dv['gm_bwd']*1000, linestyle='--', color='r')
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
    fig, ax = plt.subplots(facecolor='white', figsize=(12,9))
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
        ax.plot(dv['Id average'][:dv.rev_point]*1000, marker='o', color='b')
        ax.plot(dv['Id average'][dv.rev_point:]*1000, marker='o', color='r')
    else:
        ax.plot(dv['Id average']*1000, marker='o', color='b')
        
    ax2.plot(dv['gm_fwd']*1000 / (1e9*params['W']*params['d']/params['L']), linestyle='--', color='b')
    ax2.plot(dv['gm_bwd']*1000 / (1e9*params['W']*params['d']/params['L']), linestyle='--', color='r')
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

def plot_output_avg(dv, params):
    '''
    called by OECT_loading functions
    
    dv = dataFrame
    '''
    
    fig, ax = plt.subplots(facecolor='white', figsize=(10,8))
    
    plt.rc('axes', linewidth=4)
    ax.tick_params(labeltop=False, labelright=False)
    ax.tick_params(axis='both', length=10, width=3, which='major',
                    bottom='on', left='on', right='on', top='on')
    ax.tick_params(axis='both', length=6, width=3, which='minor',
                    bottom='on', left='on', right='on', top='on')
    plt.rcParams.update({'font.size': 24, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})

    ax.plot(dv*1000, marker='o')

    ax.set_ylabel('Ids Current (mA)', fontweight='bold', fontname='Arial')
    ax.set_xlabel('Vds Voltage (V)', fontweight='bold', fontname='Arial')

    xminor = AutoMinorLocator(4)
    ax.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(xminor)

    plt.title('Output Average', y=1.05) 

    return fig

def plot_outputs(dv, leakage=False):
    '''
    dv : OECT class object
    '''
    
    fig, ax = plt.subplots(facecolor='white', figsize=(12,8))
    
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
        ax.plot(dv.outputs.index, dv.outputs[k]*1000,
                 linewidth=2, marker = next(mk), markersize = 7)
        
        if leakage:
            ax2.plot(dv.outputs.index, dv.output_raw[k]['I_G (A)']*1000,
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

def plot_transfers_gm(dv, gm_plot=True, leakage=False):

    fig, ax1 = plt.subplots(facecolor='white', figsize=(10,8))
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
            ax1.plot(dv.transfers[k][:dv.rev_point]*1000,
                     linewidth=2, marker= next(mk), markersize=7, color='b')
            ax1.plot(dv.transfers[k][dv.rev_point:]*1000,
                     linewidth=2, marker= next(mk), markersize=7, color='r')
        
        if leakage:
            
            ax1.plot(dv.transfers.index, dv.transfer_raw[k]['I_G (A)']*1000,
                     linewidth=1, linestyle='--')

    markers = ['o', 's', '^', 'd', 'x']
    mk = cycle(markers)
   
    if gm_plot:
        for k in dv.gms_fwd:
            ax2.plot(dv.gms_fwd[k].index, dv.gms_fwd[k]*1000, 'b--', linewidth=2)
        for k in dv.gms_bwd:
            ax2.plot(dv.gms_bwd[k].index, dv.gms_bwd[k]*1000, 'r--', linewidth=2)
            
    ax1.set_ylabel('Ids Current (mA)', fontweight='bold',
                   fontsize=18, fontname='Arial')
    ax2.set_ylabel('Transconductance (mS)', rotation=-90, labelpad=20,
                   fontweight='bold', fontname='Arial', fontsize=18)
    ax1.set_xlabel('Vgs Voltage (V)', fontweight='bold', fontname='Arial',
                   fontsize = 18)

    xminor = AutoMinorLocator(4)
    ax1.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax1.yaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax2.yaxis.set_minor_locator(xminor)

    plt.title(dv.folder, y=1.05)
    
    return fig
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:17:40 2017

@author: Raj
"""

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import OECT
from itertools import cycle

""" PLOTTING FUNCTIONS """
def plot_transfer(dv):

    plt.figure()
    plt.plot(dv.transfer.index, dv.transfer)

def plot_transfer_avg(dv):
    
    fig, ax = plt.subplots(facecolor='white', figsize=(10,8))
    ax2 = ax.twinx()
    
    plt.rc('axes', linewidth=4)
    ax.tick_params(labeltop=False, labelright=False)
    ax.tick_params(axis='both', length=10, width=3, which='major',
                    bottom='on', left='on', right='on', top='on')
    ax.tick_params(axis='both', length=6, width=3, which='minor',
                    bottom='on', left='on', right='on', top='on')
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})

    ax.plot(dv.index.values, dv['avg'].values*1000, marker='o')
    ax2.plot(dv.index.values, dv['gm']*1000, linestyle='--')
    ax2.set_ylabel('Transconductance (mS)', rotation=-90, labelpad=20,
                   fontweight='bold', fontname='Arial', fontsize=18)

    ax.set_ylabel('Ids Current (mA)', fontweight='bold', fontname='Arial')
    ax.set_xlabel('Vgs Voltage (V)', fontweight='bold', fontname='Arial')

    xminor = AutoMinorLocator(4)
    ax.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(xminor)

    plt.title('Average', y=1.05) 

    return fig

def plot_output_avg(dv):
    
    fig, ax = plt.subplots(facecolor='white', figsize=(10,8))
    
    plt.rc('axes', linewidth=4)
    ax.tick_params(labeltop=False, labelright=False)
    ax.tick_params(axis='both', length=10, width=3, which='major',
                    bottom='on', left='on', right='on', top='on')
    ax.tick_params(axis='both', length=6, width=3, which='minor',
                    bottom='on', left='on', right='on', top='on')
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})

    ax.plot(dv.index.values, dv.values*1000, marker='o')

    ax.set_ylabel('Ids Current (mA)', fontweight='bold', fontname='Arial')
    ax.set_xlabel('Vds Voltage (V)', fontweight='bold', fontname='Arial')

    xminor = AutoMinorLocator(4)
    ax.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(xminor)

    plt.title('Output Average', y=1.05) 

    return fig

def plot_outputs(dv, leakage=False):

    fig, ax = plt.subplots(facecolor='white', figsize=(12,8))
    
    if leakage:
        ax2 = ax.twinx()
    
    plt.rc('axes', linewidth=4)
    ax.tick_params(labeltop=False, labelright=False)
    ax.tick_params(axis='both', length=10, width=3, which='major',
                    bottom='on', left='on', right='on', top='on', labelsize=18)
    ax.tick_params(axis='both', length=6, width=3, which='minor',
                    bottom='on', left='on', right='on', top='on', labelsize=18)
    plt.rcParams.update({'font.size': 18, 'font.weight': 'bold',
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
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold',
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

        ax1.plot(dv.transfers.index, dv.transfers[k]*1000,
                 linewidth=2, marker= next(mk), markersize=7)
        
        if leakage:
            
            ax1.plot(dv.transfers.index, dv.transfer_raw[k]['I_G (A)']*1000,
                     linewidth=1, linestyle='--')

    markers = ['o', 's', '^', 'd', 'x']
    mk = cycle(markers)
   
    if gm_plot:
        for k in dv.gms_fwd:
            ax2.plot(dv.gms_fwd[k].index, dv.gms_fwd[k]*1000, '--', linewidth=2)
            
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

def plot_transfer_gms_total(dv):

    fig, ax1 = plt.subplots(facecolor='white', figsize=(10,6))
    ax2 = ax1.twinx()

    plt.rc('axes', linewidth=4)
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})

    ax1.tick_params(axis='both', length=10, width=3, which='major', top='on')
    ax1.tick_params(axis='both', length=6, width=3, which='minor', top='on')
    ax2.tick_params(axis='both', length=10, width=3, which='major')
    ax2.tick_params(axis='both', length=6, width=3, which='minor')

    ax1.plot(dv.transfer.index, dv.transfer*1000, 'bs--',
             linewidth=2, markersize=7)

    ax2.plot(dv.gm_fwd.index, dv.gm_fwd*1000, 'r', linewidth=2)

    ax1.set_ylabel('Ids Current (mA)', fontweight='bold',
                   fontsize=14, fontname='Arial')
    ax2.set_ylabel('Transconductance (mS)', rotation=-90, labelpad=20,
                   fontweight='bold', fontname='Arial', fontsize=14)
    ax1.set_xlabel('Vgs Voltage (V)', fontweight='bold', fontname='Arial',
                   fontsize = 14)

    xminor = AutoMinorLocator(4)
    ax1.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax1.yaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax2.yaxis.set_minor_locator(xminor)

    plt.title(dv.folder, y=1.05)

    if any(dv.gm_bwd.values):

        ax2.plot(dv.gm_bwd.index, dv.gm_bwd*1000, 'g', linewidth=2)
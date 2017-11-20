# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:17:40 2017

@author: Raj
"""

from matplotlib import pyplot as plt
import OECT


""" PLOTTING FUNCTIONS """
def plot_transfer(dv):

    plt.figure()
    plt.plot(dv.transfer.index, dv.transfer)

def plot_outputs(dv):

    plt.figure(facecolor='white', figsize=(8,6))
    plt.rc('axes', linewidth=4)
    plt.tick_params(labeltop=False, labelright=False)
    plt.tick_params(axis='both', length=10, width=3, which='major',
                    bottom='on', left='on', right='on', top='on')
    plt.tick_params(axis='both', length=6, width=3, which='minor',
                    bottom='on', left='on', right='on', top='on')
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})

    markers = ['o', 's', '^', 'd', 'x']
    mk = cycle(markers)

    for k in dv.outputs.columns:
        plt.plot(dv.outputs.index, dv.outputs[k]*1000,
                 linewidth=2, marker = mk.next(), markersize = 7)

    plt.legend(labels=dv.Vg_labels, frameon=False,
               fontsize=16, loc=4)
    plt.ylabel('Ids Current (mA)', fontweight='bold', fontname='Arial')
    plt.xlabel('Vds Voltage (V)', fontweight='bold', fontname='Arial')

    xminor = AutoMinorLocator(4)
    plt.axes().xaxis.set_minor_locator(xminor)

    yminor = AutoMinorLocator(4)
    plt.axes().yaxis.set_minor_locator(yminor)

    plt.title(dv.folder, y=1.05)

def plot_transfer_gm(dv):

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
#    ax1.plot(dv.spline.index, dv.spline*1000, 'gs--',
#             linewidth=2, markersize=7)
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

        print 'a'
        ax2.plot(dv.gm_bwd.index, dv.gm_bwd*1000, 'g', linewidth=2)
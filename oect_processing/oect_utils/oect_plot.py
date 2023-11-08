# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:17:40 2017

@author: Raj
"""

import matplotlib.cbook
import numpy as np
import oect_processing as oectp
# import pyqtgraph as pg
import os
import warnings
from itertools import cycle
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

try:  # old matplotlib deprecation errors
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
except:
    pass
import pandas as pd

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


def plot_uC(dv, pg_graphs=[None, None], label='', savefig=True, axlin=None,
            axlog=None, fit=True, dot_color='r', average=False, **kwargs):
    """
    :param dv: dict of parameters needed for plotting
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
    :type dv: dict

    :param pg_graphs: UI graphs on which to plot.
        pg_graphs[0] is linear plot
        pg_graphs[1] is log plot
    :type pg_graphs: array of PlotItem
        
    :param dot_color: color to use when plotting on pg_graphs
    :type dot_color: QColor
        
    :param label: For saving files, incldues this in the filename
    :type label: str, optional
        
    :param savefig: Whether to save the figures
    :type savefig: bool, optional
        
    :param ax: Plot on an existing axis
    :type ax: matplotlib axes object, optional
        
    :param fit: Plot the best fit line or not
    :type fit: bool, optional
        
    :param average: Plot only the average gms
    :type average: bool, optional
        
    :param kwargs: Standard plotting params, e.g. {'color': 'b'}
        Default is size 10 blue squares 
    :type kwargs: dict, optional
    
    :returns: list [axlin, axlog, fig]
        WHERE
        [type] axlin is...
        [type] axlog is...
        [type] fig is...
        
    """
    if isinstance(dv, dict):
        WdL = dv['WdL']
        Vg_Vt = dv['Vg_Vt']
        uC = dv['uC']
        uC_0 = dv['uC_0']
        gms = dv['gms']
    elif 'OECTDevice' in str(type(dv)):
        WdL = dv.WdL
        Vg_Vt = dv.Vg_Vt
        uC = dv.uC
        uC_0 = dv.uC_0
        gms = dv.gms

    if average:
        df = pd.DataFrame(index=WdL)
        df['gms'] = gms
        df['Vg_Vt'] = Vg_Vt
        df = df.groupby(df.index).mean()
        WdL = df.index.values
        gms = df['gms'].values.flatten()
        Vg_Vt = df['Vg_Vt'].values

    if savefig:

        if isinstance(dv, dict):
            if 'folder' in dv:
                path = dv['folder']
        elif 'OECTDevice' in str(type(dv)):
            path = dv.path
        else:
            path = os.getcwd()

    ##### Linear Plot
    if not np.any(axlin):
        fig, axlin = plt.subplots(nrows=1, facecolor='white', figsize=(9, 6))

    params = {'color': 'b', 'markersize': 10, 'marker': 's', 'linestyle': ''}

    for k in kwargs:
        params[k] = kwargs[k]
    if 'ax' in params:
        axlin = params['ax']
        del params['ax']

    plt.rcParams.update({'font.size': 24, 'font.weight': 'bold',
                         'font.sans-serif': 'Arial'})
    plt.rc('axes', linewidth=4)
    axlin.tick_params(labeltop=False, labelright=False)
    axlin.tick_params(axis='both', length=14, width=3, which='major',
                      bottom='on', left='on', right='on', top='on')
    axlin.tick_params(axis='both', length=10, width=3, which='minor',
                      bottom='on', left='on', right='on', top='on')

    axlin.plot(np.abs(WdL * Vg_Vt) * 1e2, gms * 1000, **params)

    # plot on linear pg_graph
    if (pg_graphs[0]):
        pg_graphs[0].setTitle('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
        folder_name = os.path.basename(dv['folder'])
        for i in range(len(Vg_Vt)):  # ensure log plot won't throw error
            if Vg_Vt[i] == 0: Vg_Vt[i] = 10e-10
            if gms[i] == 0: gms[i] = 10e-10
        pg_graphs[0].plot(np.abs(WdL * Vg_Vt) * 1e2, gms * 1000, pen=None, symbolBrush=dot_color, symbol='o',
                          name=folder_name)

    axlin.set_xlabel('Wd/L * (Vg-Vt) (cm*V)')
    axlin.set_ylabel('gm (mS)')
    # ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    axlin.set_title('uC* = ' + str(np.round(uC_0 * 1e-2, 2)) + ' F/cm*V*s')
    plt.tight_layout()

    if savefig:
        fig = plt.gcf()
        fig.savefig(path + r'\scaling_uC' + label + '.tif', format='tiff')

    if fit:
        # create x-axis for fits
        _xl = np.argmin(WdL * Vg_Vt)
        _xh = np.argmax(WdL * Vg_Vt)
        Wd_L_fitx = np.arange(WdL[_xl] * Vg_Vt[_xl], WdL[_xh] * Vg_Vt[_xh], 1e-9)
        # ax.plot(Wd_L_fitx * 1e2, (uC[1] * Wd_L_fitx + uC[0]) * 1000, 'k--')
        axlin.plot(Wd_L_fitx * 1e2, (uC_0[0] * Wd_L_fitx) * 1000, 'r--')

        axlin.set_title('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
        axlin.tick_params(axis='both', length=17, width=3, which='major',
                          bottom='on', left='on', right='on', top='on')
        axlin.tick_params(axis='both', length=10, width=3, which='minor',
                          bottom='on', left='on', right='on', top='on')
        plt.tight_layout()

        print('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')

        if savefig:
            fig.savefig(path + r'\scaling_uC_+fit' + label + '.tif', format='tiff')

    #### Now the Log plot
    if not np.any(axlog):
        fig, axlog = plt.subplots(nrows=1, facecolor='white', figsize=(9, 6))

    params = {'color': 'b', 'markersize': 10, 'marker': 's', 'linestyle': ''}
    for k in kwargs:
        params[k] = kwargs[k]
        # fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))
    axlog.set_xscale('log')
    axlog.set_yscale('log')
    axlog.plot(np.abs(WdL * Vg_Vt) * 1e2, gms * 1000, **params)

    # plot on log pg_graph
    if (pg_graphs[1]):
        pg_graphs[1].setTitle('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
        folder_name = os.path.basename(dv['folder'])
        for i in range(len(Vg_Vt)):  # ensure log plot won't through error
            if Vg_Vt[i] == 0: Vg_Vt[i] = 10e-10
            if gms[i] == 0: gms[i] = 10e-10
        pg_graphs[1].plot(np.abs(WdL * Vg_Vt) * 1e2, gms * 1000, pen=None, symbolBrush=dot_color, symbol='o',
                          name=folder_name)

    axlog.set_xlabel('$Wd/L * (Vg-Vt) (cm*V)$')
    axlog.set_ylabel('$g_m (mS)$')

    axlog.set_title('$\mu$$C*$ = ' + str(np.round(uC_0 * 1e-2, 2)) + ' F/cm*V*s')
    axlog.tick_params(axis='both', length=17, width=3, which='major',
                      bottom='on', left='on', right='on', top='on')
    axlog.tick_params(axis='both', length=10, width=3, which='minor',
                      bottom='on', left='on', right='on', top='on')

    plt.tight_layout()

    if savefig:
        fig.savefig(path + r'\scaling_uC_loglog' + label + '.tif', format='tiff')

    # ax.plot(Wd_L_fitx * 1e2, (uC[1] * Wd_L_fitx + uC[0]), 'k--')
    if fit:
        axlog.plot(Wd_L_fitx * 1e2, (uC_0[0] * Wd_L_fitx) * 1000, 'r--')
        axlog.plot(Wd_L_fitx * 1e2, (uC[1] * Wd_L_fitx + uC[0]) * 1000, 'g--')

        plt.tight_layout()

        if savefig:
            fig.savefig(path + r'\scaling_uC_loglog_+fit' + label + '.tif', format='tiff')

    return [axlin, axlog, fig]


def plot_transfers_gm(dv, gm_plot=True, leakage=False):
    ''' 
    For plotting transfer and gm on the same plot for one pixel
    
    :param dv:
    :type dv:
    
    :param gm_plot:
    :type gm_plot: bool
    
    :param leakage:
    :type leakage: bool
    
    :returns: fig
    :rtype:
    '''

    fig, ax1 = plt.subplots(facecolor='white', figsize=(10, 8))
    ax2 = ax1.twinx()

    plt.rc('axes', linewidth=4)
    plt.rcParams.update({'font.size': 24, 'font.weight': 'bold',
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

    ax1.set_ylabel('$I_{ds}  (mA)$', fontweight='bold',
                   fontsize=18, fontname='Arial')
    ax2.set_ylabel('$g_m (mS)$', rotation=-90, labelpad=20,
                   fontweight='bold', fontname='Arial', fontsize=24)
    ax1.set_xlabel('$V_{gs} (V)$', fontweight='bold', fontname='Arial',
                   fontsize=24)

    xminor = AutoMinorLocator(4)
    ax1.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax1.yaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax2.yaxis.set_minor_locator(xminor)

    plt.title(dv.folder.split('\\')[-1], y=1.05)

    return fig


def plot_outputs(dv, leakage=False, direction='both', sort=False):
    '''
    :param dv:
    :type dv: OECT class object
    
    :param leakage: Show the Gate leakage current on right axis
    :type leakage : bool, optional
        
    :param direction: Plot only the specified direction or both
        'fwd', 'bwd', or 'both'
    :type direction: str
        
    :param sort: Whether to plot from lowest to highest Vg
    :type sort: bool, optional
        
    :returns:
    :rtype:
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

    nm = {}
    nms = []
    # Plot by sorted by Vg
    if sort:
        for c in dv.outputs.columns:
            if direction in c:
                nm[float(c[:-4])] = dv.outputs[c].values
                nms.append(float(c[:-4]))
        nms = np.sort(nms)
        for n in nms:
            ax.plot(dv.outputs.index, nm[n] * 1000,
                    linewidth=2, marker=next(mk), markersize=8)

        ax.legend(labels=nms, frameon=False,
                  fontsize=16, loc=4)

    else:
        for k in dv.outputs.columns:

            if direction == 'both' or direction in k:
                ax.plot(dv.outputs.index, dv.outputs[k] * 1000,
                        linewidth=2, marker=next(mk), markersize=8)

                if leakage:
                    ax2.plot(dv.outputs.index, dv.output_raw[k]['I_G (A)'] * 1000,
                             linewidth=1, linestyle='--')

        ax.legend(labels=dv.Vg_labels, frameon=False,
                  fontsize=16, loc=4)
    ax.set_ylabel('$I_{ds} (mA)$', fontweight='bold', fontname='Arial', fontsize=24)
    ax.set_xlabel('$V_{ds}  (V)$', fontweight='bold', fontname='Arial', fontsize=24)

    if leakage:
        ax2.set_ylabel('Gate leakage (mA)', fontweight='bold', fontname='Arial',
                       fontsize=18, rotation=-90)

    xminor = AutoMinorLocator(4)
    ax.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(xminor)

    plt.title(dv.folder.split('\\')[-1], y=1.05)

    return fig


def plot_output_avg(dv):
    '''
    called by OECT_loading functions
    
    :param dv:
    :type dv: dataFrame
    
    :returns:
    :rtype:
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

    ax.set_ylabel('$I_ds$ Current (mA)', fontweight='bold', fontname='Arial')
    ax.set_xlabel('$V_ds$ Voltage (V)', fontweight='bold', fontname='Arial')

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
    
    :param dv:
    :type dv:
    
    :param Wd_L:
    :type Wd_L:
    
    :param label:
    :type label: str
    
    :returns:
    :rtype:
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

    ax.set_ylabel('$I_ds$ Current (mA)', fontweight='bold', fontname='Arial')
    ax.set_xlabel('$V_gs$ Voltage (V)', fontweight='bold', fontname='Arial')

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

    ax2.set_ylabel('Norm $g_m$ (mS/nm)', rotation=-90, labelpad=20,
                   fontweight='bold', fontname='Arial', fontsize=18)

    ax.set_ylabel('$I_ds$ Current (mA)', fontweight='bold', fontname='Arial')
    ax.set_xlabel('$V_gs$ Voltage (V)', fontweight='bold', fontname='Arial')

    xminor = AutoMinorLocator(4)
    ax.xaxis.set_minor_locator(xminor)
    xminor = AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(xminor)
    plt.title('Average', y=1.05)
    plt.tight_layout()

    return fig

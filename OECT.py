# -*- coding: utf-8 -*-
"""
OECT.py: Contains OECT class for processing transistor data.

Created on Tue Oct 10 17:13:07 2017

__author__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
@author: Raj
"""

import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import interpolate as spi
from scipy import signal as sps
import numpy as np
from itertools import cycle


def loadOECT(path):
    """
    Wrapper function for processing OECT data

    USAGE:
        device1 = loadOECT(folder_name)

    """

    device = OECT(path)
    device.loaddata()
    device.all_outputs()
    device.calc_gms()

    plot_transfer_gm(device)
    plot_outputs(device)

    return device


class OECT(object):
    """
    Attributes
    ----------
    output : dict
        dict of DataFrames
        Each DataFrame is Id-Vd, with index of DataFrame set to Vd.
        All other columns removed (Id-error, Ig, Ig-error)
    output_raw : dict
        dict of DataFrames
        same as output except columns maintained
    outputs : DataFrame
        Single DataFrame of all outputs in one file.
        Assumes all data taken on same Vd range (as during an experiment)
    transfer : DataFrame
        DataFrame of Id-Vg, with index of DataFrame set to Vg
        All other columns removed (Ig-error)
    transfer_raw : DataFrame
        DataFrame of Id-Vg, with index of DataFrame set to Vg
        same as transfer except all columns maintained
    Vg_array : list of str
        list of gate voltages (Vg) used during Id-Vd sweeps
    Vg_labels: list of floats
        list of gate voltages (Vg) used during Id-Vd sweeps for plot labels
    transfer_avgs : int
        averages taken per point in transfer curve
    folder : string
        path to data folder on the computer
    gm_fwd : DataFrame
        Transconductance for forward sweep (in Siemens)
    gm_bwd : DataFrame
        Transconductance for reverse sweep (in Siemens)
    """

    def __init__(self, folder):

        self.output = {}
        self.output_raw = {}
        self.outputs = pd.DataFrame()

        self.transfer = []
        self.transfer_raw = []
        self.transfers = pd.DataFrame()

        self.Vg_array = []
        self.Vg_labels = []
        self.transfer_avgs = 1
        self.folder = folder

        self.gm_fwd = pd.DataFrame()
        self.gm_bwd = pd.DataFrame()
        self.gms_fwd = pd.DataFrame()
        self.gms_bwd = pd.DataFrame()

        self.num_outputs = 0
        self.num_transfers = 0


    def transfer_curve(self, path):
        """Loads Id-Vg transfer curve from a path"""
        self.transfer_raw = pd.read_csv(path, delimiter='\t',
                                        skipfooter=3, engine='python')
        self.transfer = self.transfer_raw
        self.transfer = self.transfer.drop(['I_DS Error (A)', 'I_G (A)',
                                            'I_G Error (A)'], 1)
        self.transfer = self.transfer.set_index('V_G')

        h = open(path)
        for line in h:

            if 'V_DS' in line:
                self.transfer_Vd = line.split()[-1]
            if 'Averages' in line:
                self.transfer_avgs = line.split()[-1]

        h.close()

    def calc_gm(self):
        """
        Calculates single gm curve in milli-Siemens
        Splits data into "forward" and "backward"
        Assumes curves taken neg to positive Vg
        """

        v = np.array(self.transfer.index)
        i = np.array(self.transfer.values)

        # creates resample voltage range for smoothed gm splines
        mx = np.argmax(v)
        vl_lo = np.arange(v[0], v[mx], 0.01)

        #Savitsky-Golay method
#        gmt = sps.savgol_filter(i[0:mx], 25, 1, deriv=1,
#                                       delta=v[2]-v[1])
#
#        self.gm_fwd = pd.DataFrame(data = gmt, index = v[0:mx])

        #univariate spline method
        s = 1e-7
        if v[2] - v[1] > 0.01:
            s = 1e-15
        
        funclo = spi.UnivariateSpline(v[0:mx], i[0:mx], k=4, s=s)
        gml = funclo.derivative()
        self.gm_fwd = pd.DataFrame(data=gml(vl_lo),
                                   index=vl_lo)
        self.spline = pd.DataFrame(data=funclo(vl_lo),
                                   index=vl_lo)

        # if backward sweep exists
        if mx != len(v)-1:

            vl_hi = np.arange(v[mx], v[-1], -0.01)
            funchi = spi.UnivariateSpline(v[mx:], i[mx:], k=4, s=s)
            gmh = funchi.derivative()
            self.gm_bwd = pd.DataFrame(data=gmh(vl_hi),
                                       index=vl_hi)
        else:

            self.gm_bwd = pd.DataFrame()

    def calc_gms(self):
        """
        Calculates all the gms in the set of data.
        Assigns each one to gm_fwd (forward) and gm_bwd (reverse)
        """

        for i in self.transfers:

            self.transfer = self.transfers[i]
            self.calc_gm()

            if self.gms_fwd.values.any():

                self.gms_fwd[i] = self.gm_fwd

            else:
                self.gms_fwd = self.gm_fwd


            if any(self.gm_bwd):

                if self.gms_bwd.values.any():
                    self.gms_bwd[i] = self.gm_bwd

                else:
                    self.gms_bwd = self.gm_bwd

        return

    def output_curve(self, path):
        """Loads Id-Vd output curves from a folder as Series in a list"""

        V = '0'
        h = open(path)
        for line in h:

            if 'V_G' in line:
                V = line.split()[-1]

        h.close()

        op = pd.read_csv(path, delimiter='\t', skipfooter=3, engine='python')
        self.output[V] = op
        self.output_raw[V] = op
        self.output[V] = self.output[V].drop(['I_DS Error (A)', 'I_G (A)',
                                              'I_G Error (A)'], 1)
        self.output[V] = self.output[V].set_index('V_DS')
        self.Vg_array.append(V)

    def all_outputs(self):
        """
        Creates a single dataFrame with all output curves
        This assumes that all data were taken at the same Vds range
        """

        self.Vg_labels = []  # corrects for labels below
        for op in self.output:

            self.Vg_labels.append(float(op))
            self.outputs[op] = self.output[op]['I_DS (A)'].values
            self.outputs = self.outputs.set_index(self.output[op].index)

        self.num_outputs = len(self.outputs.columns)

        return

    def all_transfers(self, path):

        V = len(self.transfers.columns)

        op = pd.read_csv(path, delimiter='\t', skipfooter=3, engine='python')
        df = op
        df = df.drop(['I_DS Error (A)','I_G (A)', 'I_G Error (A)'],1)
        df = df.set_index('V_G')

        if any(self.transfers.columns):
            self.transfers[V] = df
            return

        self.transfers = df
        self.transfers.columns

        return

    def loaddata(self):
        """Loads transfer and output files from a folder"""

        filelist = os.listdir(self.folder)
        files = [os.path.join(self.folder, name)
                 for name in filelist if name[-3:] == 'txt']

        for t in files:

            if 'transfer' in t:
                self.all_transfers(t)

            elif 'output' in t:
                self.output_curve(t)

        self.num_transfers = len(self.transfers.columns)
        self.num_outputs = len(self.outputs.columns)

        self.transfers = self.transfers.rename(columns={'I_DS (A)': 0})

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

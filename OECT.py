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

from scipy import interpolate as spi
from scipy import signal as sps
import numpy as np

import os

import OECT_plotting 

def load_avg(path, thickness = 30e-9):
    '''
    averages data in this particular path (for folders 'avg')
    
    path should be to folder '.../avg' 
    
    thickness  of the film
    '''

    params = {'W': 100e-6, 'L': 100e-6, 'd': thickness}
    
    filelist = os.listdir(path)
    pixel_names = ['01', '02', '03', '04']
    
    pixels = {}
    for k in filelist:
        if k not in pixel_names:
            filelist.remove(k)
    
    paths = [os.path.join(path, name) for name in filelist]

    for p, f in zip(paths, filelist):
        
        dv = loadOECT(p, params)
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
    Id_Vg = Id_Vg.set_index(first_pxl.transfers.index)
    
    # average Id-Vd at max Vd
    Id_Vd = []

    # finds column corresponding to lowest voltage (most doping), but these are strings
    idx = np.argmin(np.array([float(i) for i in first_pxl.outputs.columns]))
    volt = first_pxl.outputs.columns[idx]
    
    for dv in pixels:
    
        if not any(Id_Vg):
            
            Id_Vd = pixels[dv].outputs[volt].values
        
        else:   
            
            Id_Vd += pixels[dv].outputs[volt].values
   
    Id_Vd /= len(pixels)
    Id_Vd = pd.DataFrame(data= Id_Vd)
    Id_Vd = Id_Vd.set_index(pixels[list(pixels.keys())[0]].outputs[volt].index)
        
    return pixels, Id_Vg, Id_Vd


def uC_scale(path, thickness=30e-9):
    '''
    01 = 2000/20
    02 = 1000/20
    03 = 200/20
    04 = 50/20
    
    From Lucas:
        100umx100 um on the top 4 and 2000/20 1000/20 200/20 50/20 for the bottom row
    
    '''
    
    filelist = os.listdir(path)
    pixel_names = ['01', '02', '03', '04']
    
    pixel_names = []
    for k in filelist:
        if k not in pixel_names:
            filelist.remove(k)

    params_super = {'01': {'W': 2000e-6 ,'L': 20e-6, 'd': thickness},
                    '02': {'W': 1000e-6, 'L': 20e-6, 'd': thickness},
                    '03': {'W': 200e-6, 'L': 20e-6, 'd': thickness},
                    '04': {'W': 50e-6, 'L': 20e-6, 'd': thickness}
                    }

    paths = [os.path.join(path, name) for name in filelist]

    

    return

def loadOECT(path, params, gm_plot=True):
    """
    Wrapper function for processing OECT data

    params = {W: , L: , d: } for W, L, d of device

    USAGE:
        device1 = loadOECT(folder_name)
        

    """

    device = OECT(path, params)
    device.loaddata()
    device.calc_gms()
    
    scaling = params['W'] * params['d']/params['L']
    
    for key in device.gms_fwd:
        print(key,':', np.max(device.gms_fwd[key].values)/scaling, 'S/m'  )

    fig = OECT_plotting.plot_transfers_gm(device, gm_plot=gm_plot)
    fig.savefig(path+r'\transfer.tif', format='tiff')

    fig = OECT_plotting.plot_outputs(device)
    fig.savefig(path+r'\output.tif', format='tiff')

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
    transfer : dict
        dict of DataFrames
        DataFrame of Id-Vg, with index of DataFrame set to Vg
        All other columns removed (Ig-error)
    transfer_raw : dict
        dict of DataFrames
        DataFrame of Id-Vg, with index of DataFrame set to Vg
        same as transfer except all columns maintained
    transfers : DataFrame
        single dataFrame with all transfer curves
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

    def __init__(self, folder, params={}):

        self.output = {}
        self.output_raw = {}
        self.outputs = pd.DataFrame()

        self.transfer = {}
        self.transfer_raw = {}
        self.transfers = pd.DataFrame()

        self.Vg_array = []
        self.Vg_labels = []
        self.transfer_avgs = 1
        self.folder = folder

        self.gm_fwd = {}
        self.gm_bwd = {}
        self.gms_fwd = {}
        self.gms_bwd = {}

        self.num_outputs = 0
        self.num_transfers = 0
        
        self.W = params['W']
        self.L = params['L']
        self.d = params['d']
        

    def calc_gm(self, df):
        """
        Calculates single gm curve in milli-Siemens
        Splits data into "forward" and "backward"
        Assumes curves taken neg to positive Vg
        
        df = dataframe 
        """

        v = np.array(df.index)
        i = np.array(df.values)

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
        
#        funclo = spi.UnivariateSpline(v[0:mx], i[0:mx], k=4, s=s)
#        gml = funclo.derivative()
#        gm_fwd = pd.DataFrame(data=gml(vl_lo),
#                                   index=vl_lo)
#        self.spline = pd.DataFrame(data=funclo(vl_lo),
#                                   index=vl_lo)

#        # if backward sweep exists
#        if mx != len(v)-1:
#
#            vl_hi = np.arange(v[mx], v[-1], -0.01)
#            funchi = spi.UnivariateSpline(v[mx:], i[mx:], k=4)
#            gmh = funchi.derivative()
#            gm_bwd = pd.DataFrame(data=gmh(vl_hi),
#                                       index=vl_hi)
        funclo = np.polyfit(v[0:mx], i[0:mx], 8)
        gml = np.gradient(np.polyval(funclo, v[0:mx]), (v[2]-v[1]))
        gm_fwd = pd.DataFrame(data=gml, index=v[0:mx])

        if mx != len(v)-1:
            vl_hi = np.arange(v[mx], v[-1], -0.01)
            funchi = np.polyfit(v[mx:], i[mx:], 8)
            gmh = np.gradient(np.polyval(funchi, v[mx:]),  (v[2]-v[1]))
            gm_bwd = pd.DataFrame(data=gmh, index=v[mx:])

        else:

            gm_bwd = pd.DataFrame()
            
        return gm_fwd, gm_bwd

    def calc_gms(self):
        """
        Calculates all the gms in the set of data.
        Assigns each one to gm_fwd (forward) and gm_bwd (reverse)
        """

        for i in self.transfer:

            self.gms_fwd[i], self.gms_bwd[i] = self.calc_gm(self.transfer[i])
            
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

            print(op)
            self.Vg_labels.append(float(op))
            self.outputs[op] = self.output[op]['I_DS (A)'].values
            self.outputs = self.outputs.set_index(self.output[op].index)

        self.num_outputs = len(self.outputs.columns)

        return

    def transfer_curve(self, path):
        """Loads Id-Vg transfer curve from a path"""
        transfer_raw = pd.read_csv(path, delimiter='\t',
                                        skipfooter=3, engine='python')

        transfer_Vd = '0'
        # Finds the parameters for this transfer Curve
        h = open(path)
        for line in h:

            if 'V_DS' in line:
                transfer_Vd = line.split()[-1]
            if 'Averages' in line:
                transfer_avgs = line.split()[-1]

        h.close()
        
        if (transfer_Vd + '_0') in self.transfer:
            c = list(self.transfer.keys())[-1]
            c = str(int(c[-1])+1)
            transfer_Vd = transfer_Vd + '_' + c
        
        else:
            transfer_Vd += '_0'
        
        self.transfer[transfer_Vd] = transfer_raw
        self.transfer_raw[transfer_Vd] = transfer_raw
        self.transfer[transfer_Vd] = self.transfer[transfer_Vd].drop(['I_DS Error (A)', 'I_G (A)', 
                                                                     'I_G Error (A)'], 1)
        self.transfer[transfer_Vd] = self.transfer[transfer_Vd].set_index('V_G')

        return

    def all_transfers(self):

        """
        Creates a single dataFrame with all transfer curves (in case more than 1)
        This assumes that all data were taken at the same Vgs range
        """
        
        self.Vd_labels = []
        
        for tf in self.transfer:
            
            self.Vd_labels.append(tf)
            self.transfers[tf] = self.transfer[tf]['I_DS (A)'].values
            self.transfers = self.transfers.set_index(self.transfer[tf].index)
        
        return
        
    def loaddata(self):
        """Loads transfer and output files from a folder"""

        filelist = os.listdir(self.folder)
        files = [os.path.join(self.folder, name)
                 for name in filelist if name[-3:] == 'txt']

        for t in files:

            if 'transfer' in t:
                self.transfer_curve(t)

            elif 'output' in t:
                self.output_curve(t)

        self.all_outputs()
        try:
            self.all_transfers()
        except:
            print('Error in transfers: not all using same indices')
        
        self.num_transfers = len(self.transfers.columns)
        self.num_outputs = len(self.outputs.columns)
        
        self.files = files
        
        return

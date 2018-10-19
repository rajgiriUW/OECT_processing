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
import re

from scipy import interpolate as spi
from scipy import signal as sps
from scipy.optimize import curve_fit as cf

import numpy as np

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
        Transconductance for forward sweep (in Siemens) as one DataFrame
    gm_bwd : DataFrame
        Transconductance for reverse sweep (in Siemens) as one DataFrame
    gms_fwd : dict
        dict of Dataframes of all forward sweep gms
    gms_bwd : dict    
        dict of Dataframes of all backward sweep gms
    Vt : float
        Threshold voltage calculated from sqrt(Id) fit
    Vts : ndarray
        Threshold voltage for forward and reverse trace, 0: forward, 1: reverse
    reverse : bool
        If a reverse trace exists
    rev_point : float
        Voltage where the Id trace starts reverse sweep
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
        self.gms_fwd = pd.DataFrame()
        self.gms_bwd = pd.DataFrame()

        self.num_outputs = 0
        self.num_transfers = 0
        
        self.Vt = np.nan
        self.Vts = np.nan
        
        self.reverse = False
        self.rev_point = np.nan
        
        # load data
        self.loaddata()

        self.W, self.L, self.d = self.get_WdL(params)

    def _calc_gm(self, df):
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
        
        if mx == 0 :
            mx = np.argmin(v)
        
        vl_lo = np.arange(v[0], v[mx], 0.01)

        #Savitsky-Golay method
#        gmt = sps.savgol_filter(i[0:mx], 25, 1, deriv=1,
#                                       delta=v[2]-v[1])
#
#        gm_fwd = pd.DataFrame(data = gmt, index = v[0:mx], columns=['gm'])
#        gm_fwd.index.name = 'Voltage (V)'

        #univariate spline method
        s = 1e-7
        
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
        gm_fwd = pd.DataFrame(data=gml, index=v[0:mx], columns=['gm'])
        gm_fwd.index.name = 'Voltage (V)'

        # if reverse trace exists
        if mx != len(v)-1:
            # vl_hi = np.arange(v[mx], v[-1], -0.01)
            
            self.reverse = True
            self.rev_point = v[mx]
            
            vl_hi = np.flip(v[mx:])
            i_hi = np.flip(i[mx:])
            funchi = np.polyfit(vl_hi, i_hi, 8)
            
            gmh = np.gradient(np.polyval(funchi, vl_hi),  (vl_hi[2]-vl_hi[1]))
            gm_bwd = pd.DataFrame(data=gmh, index=vl_hi, columns=['gm'])
            gm_bwd.index.name = 'Voltage (V)'

        else:

            gm_bwd = pd.DataFrame()
            
        return gm_fwd, gm_bwd

    def calc_gms(self):
        """
        Calculates all the gms in the set of data.
        Assigns each one to gm_fwd (forward) and gm_bwd (reverse) as a dict
        
        Creates a single dataFrame gms_fwd and another gms_bwd
        """

        for i in self.transfer:

            self.gm_fwd[i], self.gm_bwd[i] = self._calc_gm(self.transfer[i])
            
        # assemble the gms into single dataframes
        for g in self.gm_fwd:
            
            gm_fwd = self.gm_fwd[g]
            
            if not gm_fwd.empty:
            
                self.gms_fwd[g] = self.gm_fwd[g]['gm'].values
                self.gms_fwd = self.gms_fwd.set_index(self.gm_fwd[g].index)
        
        for g in self.gm_bwd:
        
            gm_bwd = self.gm_bwd[g]
            
            if not gm_bwd.empty:
                
                self.gms_bwd[g] = self.gm_bwd[g]['gm'].values
                self.gms_bwd = self.gms_bwd.set_index(self.gm_bwd[g].index)
        
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
    
    def thresh(self, negative_Vt = True):
        """
        Finds the threshold voltage by fitting sqrt(Id) vs (Vg-Vt) and finding
            x-offset
            
        negative_Vt : bool
            assumes Threshold is a negative value (typical for p-type polymers)
            
        Uses a spline to fit Id curve first
        """
        
        Vts = np.array([])
        
        # is there a forward/reverse sweep
        # lo = -0.7 to 0.3, e.g and hi = 0.3 to -0.7, e.g
        mx = np.argmax(np.array(self.transfers.index))
        v_lo = self.transfers.index
        if self.reverse:

            v_lo = self.transfers.index[:mx]
            v_hi = self.transfers.index[mx:]
        
        else:
            v_lo = self.transfers.index[:mx]
        
        # linear curve-fitting
        def line_f(x, f0, f1):
        
            return f1 + f0*x
        
        # find minimum residual through fitting a line to several found peaks
        def _min_fit(Id, V):
        
            _residuals = np.array([])
            _fits = np.array([0,0])
            mx_d2 = self._find_peak(Id, V)
            
            for m in mx_d2:
            
                fit, _ = cf(line_f, V[:m], Id[:m], bounds=([-np.inf, -np.inf], [0, np.inf]))
                _res = np.sum( np.array((Id[:m] - line_f(V[:m], fit[0], fit[1])) **2))
                _fits = np.vstack((_fits,fit))
                _residuals = np.append(_residuals, _res)
                
            _fits = _fits[1:, :]
            fit = _fits[np.argmin(_residuals),:]
            
            return fit
        
        # Find and fit at inflection between regimes
        for tf in self.transfers:
        
            # use second derivative to find inflection, then fit line to get Vt
            #univariate spline method to find Id
            Id_lo = np.sqrt(np.abs(self.transfers[tf]).values[:mx])

            # minimize residuals by finding right peak
            fit = _min_fit(Id_lo, v_lo)
            
            # fits line, finds threshold from x-intercept
            Vts = np.append(Vts,-fit[1]/fit[0]) # x-intercept
            
            if self.reverse:
                Id_hi = np.sqrt(np.abs(self.transfers[tf]).values[mx:])
                
                # so signs on gradient work
                Id_hi = np.flip(Id_hi)
                v_hi = np.flip(v_hi)
                
                fit = _min_fit(Id_hi, v_hi)
                Vts = np.append(Vts,-fit[1]/fit[0]) # x-intercept
        
        self.Vt = np.mean(Vts)
        self.Vts = Vts
        
        return
    
    def _find_peak(self, Id, Vg, negative_Vt = True):
        '''
        Uses spline to find the transition point then return it for fitting Vt
          to sqrt(Id) vs Vg
        
        Id : array
            Id vs Vg, currents
            
        Vg : array
            Id vs Vg, voltages
        
        negative_Vt : bool
            Assumes Vt is a negative voltage (typical for many p-type polymer)
        '''
        
        Id_spl = spi.UnivariateSpline(Vg, Id, k=4, s=1e-7)
        V_spl= np.arange(Vg[0], Vg[-1], 0.01)
        d2 = np.gradient(np.gradient(Id_spl(V_spl)))
          
        peaks = sps.find_peaks_cwt(d2, np.arange(1,15))
        peaks = peaks[peaks > 5] #edge errors
        
        if negative_Vt:
                
            peaks = peaks[np.where(V_spl[peaks] < 0)]
            
        else:
                
            peaks = peaks[np.where(V_spl[peaks] > 0)]
        
        # find splined index in original array
        mx_d2 = [np.searchsorted(Vg, V_spl[p]) for p in peaks]
    
        return mx_d2
    
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
    
    def get_WdL(self,params):
        '''
        Finds the electrode parameters and stores internally
        '''
        
        keys = ['W', 'L', 'd']
        vals = {}
        
        for key in keys:
            if key in params.keys():
                vals[key] = params[key]
        
        # search params in first file in this folder for missing params
        fl = self.files[0]
                
        h = open(fl)
        for line in h:
                
            if 'Width' in line and 'W' not in vals.keys():
                vals['W'] = re.findall('\d+', line)[0]
            if 'Length' in line and 'L' not in vals.keys():
                vals['L'] = re.findall('\d+', line)[0]
            if 'Thickness' in line and 'd' not in vals.keys():
                vals['d'] = re.findall('\d+', line)[0]

        h.close()

        # default thickness= 40 nm
        if 'd' not in vals.keys():
            vals['d'] = 40e-9
                
        return vals['W'], vals['L'], vals['d']
    
    def get_metadata(self):
        
        metadata = ['Width', 'Length', 'thickness',
                    'Vd', 'Vg', 'Averages']
        
        # search params in first file in this folder for missing params
        fl = self.files[0]
                
        h = open(fl)
        for line in h:
                
            if 'Width' in line:
                self.W = re.findall('\d+', line)[0]
            if 'Length' in line:
                self.L = re.findall('\d+', line)[0]
            if 'Thickness' in line:
                self.d = re.findall('\d+', line)[0]
            if 'V_DS = ' in line:
                self.Vd = re.findall('\d+', line)[0]
            if 'V_G = ' in line:
                self.Vg = re.findall('\d+', line)[0]
            if 'Averages' in line:
                self.num_avgs = re.findall('\d+', line)[0]
                
        h.close()
        
        return
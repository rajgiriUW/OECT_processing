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
import warnings

from scipy import interpolate as spi
from scipy import signal as sps
from scipy.optimize import curve_fit as cf

from deriv import gm_deriv

import numpy as np


class OECT:
    """
    OECT class for processing transistor data from a folder of text files.

    Extracts the Id-Vg (transfer), Id-Vd (output), and gm (transconductance)
    Calculates Vt

    Parameters
    ----------
    folder : string, optional
        path to data folder on the computer. Default prompts a file dialog
    params : dict, optional
        device parameters, typically Width (W), length (L), thickness (d)
    gm_method : str, optional
        For calculating gm from the transfer curve Id-Vg
        'sg' = Savitsky_golay smoothed derivative
        'raw' = raw derivative
        'poly' = 8th order polynomial fit

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
    gm_fwd : DataFrame
        Transconductance for forward sweep (in Siemens) as one DataFrame
    gm_bwd : DataFrame
        Transconductance for reverse sweep (in Siemens) as one DataFrame
    gms_fwd : dict
        dict of Dataframes of all forward sweep gms
    gms_bwd : dict    
        dict of Dataframes of all backward sweep gms
    gm_peaks : ndarray
        Peak gms calculated by taking simple peak
    Vt : float
        Threshold voltage calculated from sqrt(Id) fit
    Vts : ndarray
        Threshold voltage for forward and reverse trace, 0: forward, 1: reverse
    reverse : bool
        If a reverse trace exists
    rev_point : float
        Voltage where the Id trace starts reverse sweep


    Examples
    --------
    >>> import OECT
    >>>
    >>> path = '../device_data/pixel_01'
    >>>
    >>> device = OECT.OECT(path)
    >>> device.calc_gms()
    >>> device.thresh()
    """

    def __init__(self, folder=None, params=None, gm_method='sg'):

        if folder is None:
            folder = ''

        if params is None:
            params = {}

        self.output = {}
        self.output_raw = {}
        self.outputs = pd.DataFrame()

        self.transfer = {}
        self.transfer_raw = {}
        self.transfers = pd.DataFrame()

        self.Vg_array = []
        self.Vg_labels = []
        self.Vd_labels = []

        self.transfer_avgs = 1
        self.folder = folder

        if not folder:
            from PyQt5 import QtWidgets

            app = QtWidgets.QApplication([])
            self.folder = QtWidgets.QFileDialog.getExistingDirectory(caption='Select folder of data')
            print('Loading', self.folder)
            app.closeAllWindows()
            app.exit()

        self.gm_fwd = {}
        self.gm_bwd = {}
        self.gms_fwd = pd.DataFrame()
        self.gms_bwd = pd.DataFrame()

        if gm_method not in ['raw', 'sg', 'poly']:
            warnings.warn('Bad parameter: defaulting to Savitsky-Golay filtering')
            self.gm_method = 'sg'
        else:
            self.gm_method = gm_method

        self.num_outputs = 0
        self.num_transfers = 0

        self.Vt = np.nan
        self.Vts = np.nan

        self.reverse = False
        self.rev_point = np.nan

        # load data
        self.loaddata()

        self.W, self.L, self.d = self.get_WdL(params)
        self.WdL = self.W * self.d / self.L

        return

    @staticmethod
    def _reverse(v):
        """if reverse trace exists, return max-point index and flag"""
        mx = np.argmax(v)

        if mx == 0:
            mx = np.argmin(v)

        if mx != len(v) - 1:
            return mx, True

        return mx, False

    def _calc_gm(self, df):
        """
        Calculates single gm curve in milli-Siemens
        Splits data into "forward" and "backward"
        Assumes curves taken neg to positive Vg
        
        df = dataframe 
        """

        v = np.array(df.index)
        i = np.array(df.values)

        mx, reverse = self._reverse(v)

        vl_lo = np.arange(v[0], v[mx], 0.01)

        # sg parameters
        window = np.max([int(0.04 * self.transfers.shape[0]), 3])
        polyorder = 2

        # polynomial fit parameter
        deg = 8

        # Get gm
        gml = gm_deriv(vl_lo, i[0:mx], self.gm_method, {'window': window, 'polyorder': polyorder, 'deg': deg})

        # Assign gms
        gm_fwd = pd.DataFrame(data=gml, index=v[0:mx], columns=['gm'])
        gm_fwd.index.name = 'Voltage (V)'

        gm_peaks = np.array([])
        gm_args = np.array([])

        gm_peaks = np.append(gm_peaks, np.max(gm_fwd.values))
        gm_args = np.append(gm_args, gm_fwd.index[np.argmax(gm_fwd.values)])

        # if reverse trace exists
        if reverse:
            # vl_hi = np.arange(v[mx], v[-1], -0.01)

            self.rev_point = v[mx]

            vl_hi = np.flip(v[mx:])
            i_hi = np.flip(i[mx:])

            gmh = gm_deriv(vl_hi, i_hi, self.gm_method, {'window': window, 'polyorder': polyorder, 'deg': deg})

            gm_bwd = pd.DataFrame(data=gmh, index=vl_hi, columns=['gm'])
            gm_bwd.index.name = 'Voltage (V)'

            gm_peaks = np.append(gm_peaks, np.max(gm_bwd.values))
            gm_args = np.append(gm_args, gm_bwd.index[np.argmax(gm_bwd.values)])

        else:

            gm_bwd = pd.DataFrame()  # empty dataframe

        gm_peaks = pd.DataFrame(data=gm_peaks, index=gm_args, columns=['peak gm (S)'])

        return gm_fwd, gm_bwd, gm_peaks

    def calc_gms(self):
        """
        Calculates all the gms in the set of data.
        Assigns each one to gm_fwd (forward) and gm_bwd (reverse) as a dict
        
        Creates a single dataFrame gms_fwd and another gms_bwd
        """

        for i in self.transfer:
            self.gm_fwd[i], self.gm_bwd[i], self.gm_peaks = self._calc_gm(self.transfer[i])

        self.reverse = False
        # assemble the gms into single dataframes
        for g in self.gm_fwd:

            gm_fwd = self.gm_fwd[g]

            if not gm_fwd.empty:
                self.gms_fwd[g] = self.gm_fwd[g]['gm'].values
                self.gms_fwd = self.gms_fwd.set_index(self.gm_fwd[g].index)

        for g in self.gm_bwd:

            self.reverse = True
            gm_bwd = self.gm_bwd[g]

            if not gm_bwd.empty:
                self.gms_bwd[g] = self.gm_bwd[g]['gm'].values
                self.gms_bwd = self.gms_bwd.set_index(self.gm_bwd[g].index)

        return

    def output_curve(self, path):
        """Loads Id-Vd output curves from a folder as Series in a list"""

        V = self.Vg

        op = pd.read_csv(path, delimiter='\t', engine='python')

        # Remove junk rows
        _junk = pd.to_numeric(op['V_DS'], errors='coerce')
        _junk = _junk.notnull()
        op = op.loc[_junk]
        op = op.set_index('V_DS')
        op = op.set_index(pd.to_numeric(op.index.values))

        mx, reverse = self._reverse(op.index.values)
        idx = op.index.values[mx]

        self.Vg_array.append(V)
        Vfwd = str(V) + '_fwd'
        self.output[Vfwd] = op[:idx]
        self.output_raw[Vfwd] = op[:idx]
        self.output[Vfwd] = self.output[Vfwd].drop(['I_DS Error (A)', 'I_G (A)',
                                                    'I_G Error (A)'], 1)
        if reverse:
            Vbwd = str(V) + '_bwd'
            self.output[Vbwd] = op[idx:]
            self.output_raw[Vbwd] = op[idx:]
            self.output[Vbwd] = self.output[Vbwd].drop(['I_DS Error (A)', 'I_G (A)',
                                                        'I_G Error (A)'], 1)

    def all_outputs(self):
        """
        Creates a single dataFrame with all output curves
        This assumes that all data were taken at the same Vds range
        """

        self.Vg_labels = []  # corrects for labels below

        for op in self.output:
            self.Vg_labels.append(op)
            df = pd.DataFrame(self.output[op])
            df = df.rename(columns={self.output[op].columns[0]: op})

            if self.outputs.empty:
                self.outputs = pd.DataFrame(df)
            else:
                self.outputs = pd.concat([self.outputs, df], axis=1)

        self.num_outputs = len(self.outputs.columns)

        return

    def transfer_curve(self, path):
        """Loads Id-Vg transfer curve from a path"""
        transfer_raw = pd.read_csv(path, delimiter='\t', engine='python')

        # Remove junk rows
        _junk = pd.to_numeric(transfer_raw['V_G'], errors='coerce')
        _junk = _junk.notnull()
        transfer_raw = transfer_raw.loc[_junk]
        transfer_raw = transfer_raw.set_index('V_G')
        transfer_raw = transfer_raw.set_index(pd.to_numeric(transfer_raw.index.values))

        transfer_Vd = str(self.Vd)

        if (transfer_Vd + '_0') in self.transfer:
            c = list(self.transfer.keys())[-1]
            c = str(int(c[-1]) + 1)
            transfer_Vd = transfer_Vd + '_' + c

        else:
            transfer_Vd += '_0'

        self.transfer[transfer_Vd] = transfer_raw
        self.transfer_raw[transfer_Vd] = transfer_raw
        self.transfer[transfer_Vd] = self.transfer[transfer_Vd].drop(['I_DS Error (A)', 'I_G (A)',
                                                                      'I_G Error (A)'], 1)

        return

    def all_transfers(self):

        """
        Creates a single dataFrame with all transfer curves (in case more than 1)
        This assumes that all data were taken at the same Vgs range
        """

        for tf in self.transfer:
            self.Vd_labels.append(tf)
            self.transfers[tf] = self.transfer[tf]['I_DS (A)'].values
            self.transfers = self.transfers.set_index(self.transfer[tf].index)

        return

    def thresh(self, negative_Vt=True):
        """
        Finds the threshold voltage by fitting sqrt(Id) vs (Vg-Vt) and finding
            x-offset
            
        negative_Vt : bool
            assumes Threshold is a negative value (typical for p-type polymers)
            
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

        # Find and fit at inflection between regimes
        for tf in self.transfers:

            # use second derivative to find inflection, then fit line to get Vt
            Id_lo = np.sqrt(np.abs(self.transfers[tf]).values[:mx])

            # minimize residuals by finding right peak
            fit = self._min_fit(Id_lo - np.min(Id_lo), v_lo)

            # fits line, finds threshold from x-intercept
            Vts = np.append(Vts, -fit[1] / fit[0])  # x-intercept

            if self.reverse:
                Id_hi = np.sqrt(np.abs(self.transfers[tf]).values[mx:])

                # so signs on gradient work
                Id_hi = np.flip(Id_hi)
                v_hi = np.flip(v_hi)

                try:
                    fit = self._min_fit(Id_hi - np.min(Id_hi), v_hi)
                    Vts = np.append(Vts, -fit[1] / fit[0])  # x-intercept
                except:
                    warnings.warn('Upper gm did not find correct Vt')
                    Vts = np.append(Vts, Vts[0])

        self.Vt = np.mean(Vts)
        self.Vts = Vts

        return

    def loaddata(self):
        """Loads transfer and output files from a folder"""

        filelist = os.listdir(self.folder)
        files = [os.path.join(self.folder, name)
                 for name in filelist if name[-3:] == 'txt']

        for t in files:

            self.get_metadata(t)

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

    def get_WdL(self, params):
        """
        Finds the electrode parameters and stores internally
        """

        keys = ['W', 'L', 'd']
        vals = {}

        for key in keys:
            if key in params.keys():
                vals[key] = params[key]

        # any missing keys?
        check_files = False
        for key in keys:
            if key not in vals.keys():
                check_files = True

        if check_files:
            # search params in first file in this folder for missing params
            fl = self.files[0]

            h = open(fl)
            for line in h:

                if 'Width' in line and 'W' not in vals.keys():
                    vals['W'] = float(line.split()[-1])
                if 'Length' in line and 'L' not in vals.keys():
                    vals['L'] = float(line.split()[-1])
                if 'Thickness' in line and 'd' not in vals.keys():
                    vals['d'] = float(line.split()[-1])

            h.close()

        # default thickness= 40 nm
        if 'd' not in vals.keys():
            vals['d'] = 40e-9

        return vals['W'], vals['L'], vals['d']

    def get_metadata(self, fl):
        """ Called in load_data to extract parameters """

        metadata = ['Vd', 'Vg', 'Averages']

        # search params in first file in this folder for missing params
        h = open(fl)
        for line in h:

            if 'V_DS = ' in line:
                self.Vd = float(line.split()[-1])
            if 'V_G = ' in line:
                self.Vg = float(line.split()[-1])
            if 'Averages' in line:
                self.num_avgs = float(line.split()[-1])

        h.close()

        return

    # find minimum residual through fitting a line to several found peaks
    def _min_fit(self, Id, V):

        _residuals = np.array([])
        _fits = np.array([0, 0])
        mx_d2 = self._find_peak(Id, V)

        for m in mx_d2:
            #                Id = Id - np.min(Id) # 0-offset
            fit, _ = cf(self.line_f, V[:m], Id[:m], bounds=([-np.inf, -np.inf], [0, np.inf]))
            _res = np.sum(np.array((Id[:m] - self.line_f(V[:m], fit[0], fit[1])) ** 2))
            _fits = np.vstack((_fits, fit))
            _residuals = np.append(_residuals, _res)

        _fits = _fits[1:, :]
        fit = _fits[np.argmin(_residuals), :]

        return fit

    # linear curve-fitting
    @staticmethod
    def line_f(x, f0, f1):

        return f1 + f0 * x

    @staticmethod
    def _find_peak(Id, Vg, negative_Vt=True):
        """
        Uses spline to find the transition point then return it for fitting Vt
          to sqrt(Id) vs Vg

        Parameters
        ----------
        Id : array
            Id vs Vg, currents
        Vg : array
            Id vs Vg, voltages
        negative_Vt : bool
            Assumes Vt is a negative voltage (typical for many p-type polymer)

        Returns
        -------
        mxd2 : list
            index of the maximum transition point for threshold voltage calculation
        """

        # uses second derivative for transition point
        Id_spl = spi.UnivariateSpline(Vg, Id, k=4, s=1e-7)
        V_spl = np.arange(Vg[0], Vg[-1], 0.01)
        d2 = np.gradient(np.gradient(Id_spl(V_spl)))

        peaks = sps.find_peaks_cwt(d2, np.arange(1, 15))
        peaks = peaks[peaks > 5]  # edge errors

        if negative_Vt:

            peaks = peaks[np.where(V_spl[peaks] < 0)]

        else:

            peaks = peaks[np.where(V_spl[peaks] > 0)]

        # find splined index in original array
        mx_d2 = [np.searchsorted(Vg, V_spl[p]) for p in peaks]

        return mx_d2

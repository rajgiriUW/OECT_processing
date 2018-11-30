# -*- coding: utf-8 -*-
"""
OECT.py: Contains OECT class for processing transistor data.

Created on Tue Oct 10 17:13:07 2017

__author__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
@author: Raj
"""

import configparser
import os
import warnings

import numpy as np
import pandas as pd
from scipy import interpolate as spi
from scipy import signal as sps
from scipy.optimize import curve_fit as cf

from deriv import gm_deriv

warnings.simplefilter(action='ignore', category=FutureWarning)


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
    options : dict, optional
        processing optional parameters (for transfer curves only)
        Reverse : bool
            Whether to process the reverse trace since there's often hysteresis
        Average : bool
            Whether instead to average forward and reverse trace
            Reverse XOR Average must be true
        gm_method : str
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

    def __init__(self, folder=None, params=None, options=None):

        if folder is None:
            folder = ''

        # Data containers
        self.output = {}
        self.output_raw = {}
        self.outputs = pd.DataFrame()
        self.transfer = {}
        self.transfer_raw = {}
        self.transfers = pd.DataFrame()
        self.Vg_array = []
        self.Vg_labels = []
        self.Vd_labels = []
        self.gm_fwd = {}
        self.gm_bwd = {}
        self.gms_fwd = pd.DataFrame()
        self.gms_bwd = pd.DataFrame()

        # Data descriptors
        self.transfer_avgs = 1
        self.folder = folder
        self.num_outputs = 0
        self.num_transfers = 0

        # Threshold
        self.Vt = np.nan
        self.Vts = np.nan

        self.reverse = False
        self.rev_point = np.nan

        if not folder:
            from PyQt5 import QtWidgets

            app = QtWidgets.QApplication([])
            self.folder = QtWidgets.QFileDialog.getExistingDirectory(caption='Select folder of data')
            print('Loading', self.folder)
            app.closeAllWindows()
            app.exit()

        # load data, finds config file
        self.filelist()
        if self.config:
            _par, _opt = config_file(self.config)
        else:
            _par = []
            _opt = []

        # processing and device parameters
        self.params = {}
        self.options = {}

        if params is not None:
            for p in params:
                self.params[p] = params[p]
        for p in _par:
            self.params[p] = _par[p]

        if options is not None:
            for o in options:
                self.options[o] = options[o]
        for o in _opt:
            self.options[o] = _opt[o]

        # defaults
        if 'gm_method' not in self.options:
            self.options['gm_method'] = 'sg'
        if 'Reverse' not in self.options:
            self.options['Reverse'] = True
            self.options['Average'] = False

        self.loaddata()

        self.W, self.L = self.params['W'], self.params['L']
        if 'd' not in self.params:
            self.params['d'] = 40e-9
        self.d = self.params['d']

        self.WdL = self.W * self.d / self.L

        return

    def _reverse(self, v):
        """if reverse trace exists, return max-point index and flag"""
        mx = np.argmax(v)

        if mx == 0:
            mx = np.argmin(v)

        if self.options['Reverse']:
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
        vl_lo = v[:mx]

        # sg parameters
        window = np.max([int(0.04 * self.transfers.shape[0]), 3])
        polyorder = 2

        # polynomial fit parameter
        deg = 8

        # Get gm
        gml = gm_deriv(vl_lo, i[0:mx], self.options['gm_method'],
                       {'window': window, 'polyorder': polyorder, 'deg': deg})

        # Assign gms
        gm_fwd = pd.DataFrame(data=gml, index=v[0:mx], columns=['gm'])
        gm_fwd.index.name = 'Voltage (V)'

        gm_peaks = np.array([])
        gm_args = np.array([])

        gm_peaks = np.append(gm_peaks, np.max(gm_fwd.values))
        gm_args = np.append(gm_args, gm_fwd.index[np.argmax(gm_fwd.values)])

        # if reverse trace exists and we want to process it
        if reverse:
            # vl_hi = np.arange(v[mx], v[-1], -0.01)

            self.rev_point = v[mx]

            vl_hi = np.flip(v[mx:])
            i_hi = np.flip(i[mx:])

            gmh = gm_deriv(vl_hi, i_hi, self.options['gm_method'],
                           {'window': window, 'polyorder': polyorder, 'deg': deg})

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

            transfer = self.transfer[tf]['I_DS (A)'].values
            idx = self.transfer[tf]['I_DS (A)'].index.values
            if 'Average' in self.options and self.options['Average']:
                mx, _ = self._reverse(idx)
                idx = idx[0:mx]
                fwd = transfer[:mx]
                bwd = np.flip(transfer[mx + 1:])
                transfer = (fwd + bwd) / 2

            self.transfers[tf] = transfer
            self.transfers = self.transfers.set_index(pd.Index(idx))

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

            if self.options['Reverse']:
                Id_hi = np.sqrt(np.abs(self.transfers[tf]).values[mx:])

                # so signs on gradient work
                Id_hi = np.flip(Id_hi)
                v_hi = np.flip(v_hi)

                try:
                    fit = self._min_fit(Id_hi - np.min(Id_hi), v_hi)
                    print(fit)
                    Vts = np.append(Vts, -fit[1] / fit[0])  # x-intercept
                except:
                    warnings.warn('Upper gm did not find correct Vt')

        self.Vt = np.mean(Vts)
        self.Vts = Vts

        return

    def filelist(self):
        """ Generates list of files to process and config file"""

        filelist = os.listdir(self.folder)
        files = [os.path.join(self.folder, name)
                 for name in filelist if name[-3:] == 'txt']

        # find config file
        config = [os.path.join(self.folder, name)
                  for name in filelist if name[-10:] == 'config.cfg']

        if config:

            for f in files:

                if 'config' in f:
                    files.remove(f)

            self.config = config

        else:

            print('No config file found!')
            self.config = None

        self.files = files

        return

    def loaddata(self):
        """
        3 Steps to loading a folder of data:
            1) generate filelist for only txt files
            2) determine if config exists (newer devices)
            3) for each file in the filelist, generate a transfer curve or output curve

        """

        for t in self.files:

            self.get_metadata(t)

            if 'transfer' in t:
                self.transfer_curve(t)

            elif 'output' in t:
                self.output_curve(t)

        self.all_outputs()

        self.all_transfers()
        # try:
        #     self.all_transfers()
        # except:
        #     print('Error in transfers: not all using same indices')

        self.num_transfers = len(self.transfers.columns)
        self.num_outputs = len(self.outputs.columns)

        return

    def get_metadata(self, fl):
        """ Called in load_data to extract file-specific parameters """

        metadata = ['Vd', 'Vg', 'Averages']

        # search params in first file in this folder for missing params
        h = open(fl)
        for line in h:

            if 'V_DS = ' in line:
                self.Vd = float(line.split()[-1])
            if 'V_G = ' in line:
                self.Vg = float(line.split()[-1])

        h.close()

        return

    # find minimum residual through fitting a line to several found peaks
    def _min_fit(self, Id, V):

        _residuals = np.array([])
        _fits = np.array([0, 0])
        mx_d2 = self._find_peak(Id, V)

        # for each peak found, fits a line. Uses that to determine Vt, then residual up to that found Vt
        for m in mx_d2:
            # Id = Id - np.min(Id) # 0-offset
            fit, _ = cf(self.line_f, V[:m], Id[:m],
                        bounds=([-np.inf, -np.inf], [0, np.inf]))

            v_x = np.searchsorted(V, -fit[1] / fit[0])  # finds the Vt from this fit to determine residual
            _res = np.sum(np.array((Id[:v_x] - self.line_f(V[:v_x], fit[0], fit[1])) ** 2))
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


def config_file(cfg):
    """
    Generates parameters from supplied config file
    """
    config = configparser.ConfigParser()
    config.read(cfg)
    params = {}
    options = {}

    dim_keys = {'Width (um)': 'W', 'Length (um)': 'L', 'Thickness (nm)': 'd'}
    vgs_keys = ['Preread (ms)', 'First Bias (ms)', 'Vds (V)']
    vds_keys = ['Preread (ms)', 'First Bias (ms)', 'Output Vgs']
    opts_bools = ['Reverse', 'Average']
    opts_str = ['gm_method']

    for key in dim_keys:

        if config.has_option('Dimensions', key):
            params[dim_keys[key]] = config.getint('Dimensions', key)

    for key in vgs_keys:

        if config.has_option('Transfer', key):
            params[key] = int(config.getfloat('Transfer', key))

    for key in vds_keys:

        if config.has_option('Output', key):
            val = int(config.getfloat('Output', key))

            # to avoid duplicate keys
            if key in params:
                key = 'output_' + key
            params[key] = val

    if 'Output Vgs' in params:

        params['Vgs'] = []
        for i in range(1, params['Output Vgs'] + 1):
            nm = 'Vgs (V)\t' + str(i)

            val = config.getfloat('Output', nm)
            params['Vgs'].append(val)

    if 'Options' in config.sections():

        for key in opts_bools:

            if config.has_option('Options', key):
                options[key] = config.getboolean('Options', key)

        for key in opts_str:

            if config.has_option('Options', key):
                options[key] = config.get('Options', key)

    return params, options

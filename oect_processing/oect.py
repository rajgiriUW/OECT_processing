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
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy import interpolate as spi
from scipy import signal as sps
from scipy.optimize import curve_fit as cf

try:
    from .oect_utils.config import make_config, config_file
    from .oect_utils.deriv import gm_deriv
except:  # Jupyter
    from oect_utils.config import make_config, config_file
    from oect_utils.deriv import gm_deriv

warnings.simplefilter(action='ignore', category=FutureWarning)


class OECT:
    '''
    Processes OECT transistor data from a folder of text files.

    Transfer curve files must contain 'transfer' in the filename; output curve
    files must contain 'output'. Splits all traces into forward and reverse sweeps
    and auto-calculates threshold voltage. Uses a Savitzky-Golay smoothed derivative
    for gm by default.

    Parameters
    ----------
    folder : str, optional
        Path to the data folder. Prompts a file dialog if not provided.
    dimDict : dict, optional
        Explicit dimensions in the form
        {parentfolder: {subfolder: (W, L), ...}, ...}.
    params : dict, optional
        Device parameters (override the .cfg file):
        W : float, microns
        L : float, microns
        d : float, nanometres (converted to metres internally)
        capacitance : float, Farads
        c_star : float, F/cm^3 (volumetric capacitance from EIS)
    options : dict, optional
        Reverse : bool
            Use the reverse sweep (default True).
        Average : bool
            Average forward and reverse sweeps (mutually exclusive with Reverse).
        gm_method : str
            'sg' (Savitzky-Golay), 'raw' (np.gradient), or 'poly' (polynomial).
        overwrite : bool
            Overwrite the config file.
        V_low : bool
            Detect non-monotonic transfer curves at very negative voltages.

    Usage
    -----
    >>> import oect_processing as oect
    >>> device = oect.OECT('../device_data/pixel_01')
    >>> device.calc_gms()
    >>> device.thresh()
    >>> from oect_processing.oect_utils import oect_plot
    >>> oect_plot.plot_transfers_gm(device)
    >>> oect_plot.plot_outputs(device)

    Attributes
    ----------
    outputs : DataFrame
        All output curves (Id-Vd), assumes the same Vd range across files.
    transfers : DataFrame
        All transfer curves (Id-Vg).
    gms : DataFrame
        Transconductance for all sweeps (Siemens).
    Vts : ndarray
        Threshold voltages; element 0 = forward, 1 = reverse.
    VgVts : ndarray
        Vg at peak gm minus Vt; used in uC* calculation.
    WdL : float
        W * d / L prefactor (metres).
    c_star : float
        Volumetric capacitance (F/cm^3), user-supplied.
    capacitance : float
        Total capacitance (F), user-supplied, scaled internally to c_star.
    VgVts_spl : ndarray
        VgVts computed via spline fitting.
    output : dict
        Per-gate-voltage Id-Vd DataFrames (Id column only).
    output_raw : dict
        Per-gate-voltage Id-Vd DataFrames (all columns).
    transfer : dict
        Per-Vd Id-Vg DataFrames (Id column only).
    transfer_raw : dict
        Per-Vd Id-Vg DataFrames (all columns).
    Vg_array : list
        Gate voltages used during Id-Vd sweeps.
    Vg_labels : list
        Gate voltage labels for plotting.
    transfer_avgs : int
        Averages taken per point in the transfer curve.
    gm_fwd : dict
        Forward-sweep transconductance DataFrames.
    gm_bwd : dict
        Backward-sweep transconductance DataFrames.
    gm_peaks : DataFrame
        Peak gm values and voltages (smoothed derivative).
    peak_gm : ndarray
        Peak gm values.
    gm_peaks_spl : DataFrame
        Peak gm values and voltages (spline).
    peak_gm_spl : ndarray
        Peak gm values from spline.
    Vt : float
        Mean threshold voltage from sqrt(Id) fit.
    reverse : bool
        Whether a reverse sweep was detected.
    rev_point : float
        Voltage index where the sweep reverses.
    '''

    def __init__(self,
                 folder='',
                 dimDict={},
                 params={},
                 options={}):

        # Data containers
        self.output = {}
        self.output_raw = {}
        self.outputs = pd.DataFrame()
        self.transfer = {}
        self.transfer_raw = {}
        self.transfers = pd.DataFrame()
        self.Vg_array = []
        self.Vd_array = []
        self.Vg_labels = []
        self.Vd_labels = []
        self.gm_fwd = {}
        self.gm_bwd = {}
        self.gm_fwd_spl = {}
        self.gm_bwd_spl = {}
        self.gms = pd.DataFrame()
        self.gms_spl = pd.DataFrame()
        self.peak_gm = None

        # Data descriptors
        self.make_config = False  # if config doesn't exist, for backwards-compatibility
        self.transfer_avgs = 1
        self.folder = folder
        self.num_outputs = 0
        self.num_transfers = 0
        self.reverse = False
        self.rev_point = np.nan

        # Transconductance params
        # window/polyorder/degree are for Savitsky-Golay
        # k is for spline method
        self.gm_options = {'window': 11, 
                           'polyorder': 2, 
                           'deg': 8,
                           'k': 5}

        # Threshold
        self.Vt = np.nan
        self.Vts = np.nan

        if not folder:
            from PyQt5 import QtWidgets

            app = QtWidgets.QApplication([])
            self.folder = QtWidgets.QFileDialog.getExistingDirectory(caption='Select folder of data')
            print('Loading', self.folder)
            app.closeAllWindows()
            app.exit()

        # load data, finds config file
        self.filelist()
        _par, _opt = config_file(self.config)

        self.set_params(_par, _opt, params, options)
        self.loaddata()

        if dimDict:  # set W and L based on dictionary
            subfolder = os.path.basename(folder)
            parentFolder = os.path.dirname(folder)
            dims = dimDict[parentFolder][subfolder]
            self.W = dims[0]
            self.params['W'] = dims[0]
            self.L = dims[1]
            self.params['L'] = dims[1]
        else:
            self.W, self.L = self.params['W'], self.params['L']

        if 'd' not in self.params or not self.params['d']:
            self.params['d'] = 40e-9
        elif self.params['d'] > 1:  # wrong units
            self.params['d'] *= 1e-9
        self.d = self.params['d']

        self.WdL = self.W * self.d / self.L

        self.c_star = None
        if 'c_star' in self.params and self.params['c_star'] != None:
            self.c_star = self.params['c_star']
        elif 'capacitance' in self.params and self.params['capacitance'] != None:
            self.c_star = self.params['capacitance'] / (self.W * 1e-4 * self.L * 1e-4 * self.d)

        return

    def set_params(self, par, opt, params, options):
        '''
        Merges config-file parameters with caller-supplied overrides.

        Parameters
        ----------
        par : dict
            Parameters from the config file.
        opt : dict
            Options from the config file.
        params : dict
            Parameters passed at construction (override par).
        options : dict
            Options passed at construction (override opt).
        '''
        # processing and device parameters
        self.params = {}
        self.options = {}

        for k in [par, opt, params, options]:
            if k and not isinstance(k, dict):
                raise TypeError('Must pass a dictionary')
        # From the config file
        self.params.update(par)
        self.options.update(opt)

        # Overwrite with passed parameters
        if any(params):
            self.params.update(params)
        if any(options):
            self.options.update(options)

        # defaults
        if 'gm_method' not in self.options:
            self.options['gm_method'] = 'sg'
        if 'Reverse' not in self.options:
            self.options['Reverse'] = True
        if 'Average' not in self.options:
            self.options['Average'] = False
        if 'V_low' not in self.options:
            self.options['V_low'] = False
        if 'overwrite' not in self.options:
            self.options['overwrite'] = False

        return

    def _reverse(self, v, transfer=False):
        """
        Detects whether a reverse sweep exists and returns the inflection index.

        Parameters
        ----------
        v : array-like
            Voltage array to inspect.
        transfer : bool, optional
            If True, saves rev_point and rev_v as instance attributes.

        Returns
        -------
        mx : int
            Index of the voltage inflection point.
        reverse : bool
            True if a reverse sweep was detected.
        """

        # find inflection point where trace reverses

        # First, check if any voltages repeated (indicates both a fwd/rev sweep)
        # two ways to do this

        # a) find reverse sweep using np.allclose
        mx = np.where(np.diff(np.sign(np.diff(v))) != 0)[0]
        if not any(mx):
            reverse = False
        else:
            if len(mx) == 2:
                mx = mx[-1] + 1
            else:
                mx = mx[-1] + 2
            reverse = True

        if reverse:

            if transfer:
                self.rev_point = mx  # find inflection
                self.rev_v = v[mx]
                self.options['Reverse'] = True
                self.reverse = True

            return mx, True

        else:

            mx = len(v) - 1

            if transfer:
                self.rev_point = mx
                self.rev_v = v[mx]
                self.options['Reverse'] = False
                self.reverse = False

            return mx, False

    def calc_gms(self):
        """
        Calculates all the gms in the set of data.
        Assigns each one to gm_fwd (forward) and gm_bwd (reverse) as a dict

        Creates a single dataFrame gms_fwd and another gms_bwd
        Creates a dataFrame for gm_peaks (peak transconductance)
        Creates a dataFrame for gm_peaks_spl (peak transconductance via splines)
        """
        # Calculates the gm and gm_peak values/voltages for all transfers
        self.gm_peaks = pd.DataFrame(columns=['peak gm (S)'])
        self.gm_peaks_spl = pd.DataFrame(columns=['peak gm spline (S)'])
        for i in self.transfer:
            data = self._calc_gm(self.transfer[i])
            self.gm_fwd[i], self.gm_bwd[i] = data[0], data[1]
            self.gm_fwd_spl[i], self.gm_bwd_spl[i] = data[3], data[4]
            gm_peaks, gm_peaks_spl = data[2], data[5]
           
            if self.gm_peaks.empty:
                self.gm_peaks = gm_peaks
                self.gm_peaks_spl = gm_peaks_spl
           
            else:
                self.gm_peaks = pd.concat([self.gm_peaks, gm_peaks])
                self.gm_peaks_spl = pd.concat([self.gm_peaks_spl, gm_peaks_spl])
        
        self.gm_peaks.index.name = 'Voltage (V)'
        self.gm_peaks_spl.index.name = 'Voltage (V)'
        
        # combine all the gm_fwd and gm_bwd into a single dataframe
        def store_gm(gm_arr, gms, labels=0):
            for g in gm_arr:
                
                if not gm_arr[g].empty:
                    gm = gm_arr[g].values.flatten()
                    ix = gm_arr[g].index.values
                    
                    nm = 'gm_' + g
                    while nm in gms:
                        labels += 1
                        nm = 'gm_' + g[:-1] + str(labels)
                        
                    df = pd.Series(data = gm, index = ix)
                    try:
                        gms[nm] = df
                    except:
                        print('gm assignment error')
                        continue
        
            return gms, labels
        
        _, labels = store_gm(self.gm_fwd, self.gms, 0)
        _, _ = store_gm(self.gm_bwd, self.gms, labels)
        _, labels = store_gm(self.gm_fwd_spl, self.gms_spl, 0)
        _, _ = store_gm(self.gm_bwd_spl, self.gms_spl, labels)

        # Store and average gm values
        self.peak_gm = self.gm_peaks['peak gm (S)'].values
        self.peak_gm_spl = self.gm_peaks_spl['peak gm spline (S)'].values

        if 'Average' in self.options and self.options['Average']:
            self.gms = pd.DataFrame(self.gms.mean(1), columns=['gm_avg'])
            self.peak_gm = self.gm_peaks['peak gm (S)'].values.mean()
            self.peak_gm_spl = self.gm_peaks_spl['peak gm spline (S)'].values.mean()

        return

    def _calc_gm(self, df):
        """
        Calculates the transconductance for one transfer curve in Siemens.

        Splits into forward and backward sweeps. Assumes curves go from
        negative to positive Vg.

        Parameters
        ----------
        df : DataFrame
            Single transfer curve (Id-Vg) with voltage as index.

        Returns
        -------
        gm_fwd : DataFrame
            Forward-sweep transconductance.
        gm_bwd : DataFrame
            Backward-sweep transconductance (empty if no reverse sweep).
        gm_peaks : DataFrame
            Peak gm values and voltages (smoothed derivative).
        gm_fwd_spl : DataFrame
            Forward-sweep gm from spline.
        gm_bwd_spl : DataFrame
            Backward-sweep gm from spline.
        gm_peaks_spl : DataFrame
            Peak gm values and voltages (spline).
        """

        v = np.array(df.index)
        i = np.array(df.values).flatten()

        mx, reverse = self.rev_point, self.reverse

        vl_lo = v[:mx]

        gm_peaks = np.array([])
        gm_args = np.array([])
        gm_peaks_spl = np.array([])
        gm_args_spl = np.array([])

        # sg parameters
        self.gm_options['window'] = np.max([int(0.04 * self.transfers.shape[0]), 3])
        # polyorder = 2
        # deg = 8
        # fitparams = {'window': window, 'polyorder': polyorder, 'deg': deg}

        # Uses derivative in deriv.py file
        def get_gm(v, i, fit, options):

            gml = gm_deriv(v, i, fit, options)
            gm = pd.DataFrame(data=gml, index=v, columns=['gm'])
            gm.index.name = 'Voltage (V)'

            return gm
        
        # Gets spline for smoothed Vg and gm values
        def get_gm_spline(v, i, k=3, s=1e-7):
            
            if not all([x < y for x, y in zip(v[:], v[1:])]):
                v = v[::-1]
                i = i[::-1]
            spl = spi.UnivariateSpline(v, i, k=k, s=s)
            _v = np.arange(v[0], v[-1], 0.001)
            gml = np.gradient(spl(_v))/(_v[1]- _v[0])
            gml_df = pd.DataFrame(data=gml, index = _v, columns=['gm'])
            
            return spl, gml_df, _v
            
        # Get gm
        gm_fwd = get_gm(vl_lo, i[0:mx], self.options['gm_method'], self.gm_options)
        gm_peaks = np.append(gm_peaks, np.max(gm_fwd.values))
        gm_args = np.append(gm_args, gm_fwd.index[np.argmax(gm_fwd.values)])

        # Using spline to smooth derivative
        spl, gm_fwd_spl, v_spl = get_gm_spline(vl_lo, i[0:mx], self.gm_options['k'])
        gm_peaks_spl = np.append(gm_peaks_spl, np.max(gm_fwd_spl))
        gm_args_spl = np.append(gm_args_spl, v_spl[np.argmax(gm_fwd_spl)])

        # if reverse trace exists and we want to process it
        if reverse:
            vl_hi = np.flip(v[mx:])
            i_hi = np.flip(i[mx:])

            gm_bwd = get_gm(vl_hi, i_hi, self.options['gm_method'], self.gm_options)

            gm_peaks = np.append(gm_peaks, np.max(gm_bwd.values))
            gm_args = np.append(gm_args, gm_bwd.index[np.argmax(gm_bwd.values)])
            
            spl, gm_bwd_spl, v_spl = get_gm_spline(vl_hi, i_hi, self.gm_options['k'])
            gm_peaks_spl = np.append(gm_peaks_spl, np.max(gm_bwd_spl))
            gm_args_spl = np.append(gm_args_spl, v_spl[np.argmax(gm_bwd_spl)])

        else:

            gm_bwd = pd.DataFrame()  # empty dataframe
            gm_bwd_spl = pd.DataFrame()

        gm_peaks = pd.DataFrame(data=gm_peaks, index=gm_args, columns=['peak gm (S)'])
        gm_peaks_spl = pd.DataFrame(data=gm_peaks_spl, index=gm_args_spl, columns=['peak gm spline (S)'])
        print(gm_peaks)
        print(gm_peaks_spl)
        return gm_fwd, gm_bwd, gm_peaks, gm_fwd_spl, gm_bwd_spl, gm_peaks_spl

    def quadrant(self):

        if np.any(self.gm_peaks.index < 0):

            self.quad = 'III'  # positive voltage, positive current

        elif np.any(self.gm_peaks.index > 0):

            self.quad = 'I'  # negative voltage, negative curret=nt

        return

    def thresh(self, plot=False, c_star=None, cap=None):
        """
        Finds the threshold voltage by fitting sqrt(Id) vs Vg.

        Uses Id_sat^0.5 = sqrt(uC*/2 * Wd/L) * (Vg - Vt), fitting a line to
        find the x-intercept Vt.

        Parameters
        ----------
        plot : bool, optional
            If True, plots the sqrt(Id) trace and fit line.
        c_star : float, optional
            Volumetric capacitance (F/cm^3). Takes priority over cap.
        cap : float, optional
            Total capacitance (F), scaled internally to F/cm^3 via W*d*L.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            Only returned when plot=True.
        """

        if not hasattr(self, 'gm_peaks'):
            raise AttributeError('gm_peaks not found. Did you run calc_gms()?')

        Vts = np.array([])
        VgVts = np.array([])
        VgVts_spl = np.array([])
        mobilities = np.array([])

        if plot:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(facecolor='white')
            # plt.figure()
            ax.set_xlabel('$V_{GS}$ $Voltage (V)$')
            ax.set_ylabel('|$I_{DS}$$^{0.5}$| ($A^{0.5}$)')
            labels = []

        if c_star:
            self.c_star = c_star

        elif cap:
            vol = self.W * 1e-4 * self.L * 1e-4 * self.d  # assumes W, L in um
            self.c_star = cap / vol  # in F/cm^3

        # Find and fit at inflection between regimes
        for tf, pk, pk_spl in zip(self.transfers, self.gm_peaks.index, self.gm_peaks_spl.index):
            # use second derivative to find inflection, then fit line to get Vt
            v_lo = self.transfers[tf].index.values
            Id_lo = np.sqrt(np.abs(self.transfers[tf]).values)

            # Check for nans
            _ix = np.where(np.isnan(Id_lo) == False)
            if any(_ix[0]):
                Id_lo = Id_lo[_ix[0]]
                v_lo = v_lo[_ix[0]]

            # minimize residuals by finding right peak
            #fit = self._min_fit(Id_lo - np.min(Id_lo), v_lo)
            
            _pk_ix = self.gms.index.get_loc(pk)
            if _pk_ix == 0:
                _pk_ix = -1 #for non-reversed cases
            fit = self._min_fit(Id_lo[:_pk_ix] - np.min(Id_lo[:_pk_ix]), v_lo[:_pk_ix])

            if plot:
                plt.plot(np.sqrt(np.abs(self.transfers[tf])), 'bo-')
                v = self.transfers[tf].index.values
                v = v_lo[:_pk_ix]
                tx = np.arange(np.min(v), -fit[1] / fit[0] + 0.025, 0.0025)

                if self.quad == 'I':
                    tx = np.arange(-fit[1] / fit[0] - 0.1, np.max(v), 0.0025)

                plt.plot(tx, self.line_f(tx, *fit), 'r--')
                labels.append('{:.4f}'.format(-fit[1] / fit[0]))

            # fits line, finds threshold from x-intercept
            Vts = np.append(Vts, -fit[1] / fit[0])  # x-intercept
            VgVts = np.append(VgVts, np.abs(pk + fit[1] / fit[0]))  # Vg - Vt, + sign from -fit[1]/fit[0]
            VgVts_spl = np.append(VgVts_spl, np.abs(pk_spl + fit[1] / fit[0]))

            if self.c_star:
                mu = (-Vts[-1] * np.sqrt(0.5 * self.c_star * self.WdL * 1e2))  # 1e2 for scaling Wd/L to cm
                mu = (fit[1] / mu) ** 2
                mobilities = np.append(mobilities, mu)

        if plot:
            ax.legend(labels=labels)
            ax.axhline(0, color='k', linestyle='--')
            for v in Vts:
                ax.axvline(v, color='xkcd:periwinkle', linestyle='--')
                ax.axvline(0, color='k', linestyle='--')

        self.Vt = np.mean(Vts)
        self.Vts = Vts
        self.VgVt = np.mean(VgVts)
        self.VgVts = VgVts
        self.VgVts_spl = VgVts_spl
        self.mobilities = mobilities
        self.mobility = np.mean(mobilities)

        if plot:
            return fig, ax

        return

    # find minimum residual through fitting a line to several found peaks
    def _min_fit(self, Id, V):
        """
        Finds the best linear fit to the saturation regime by minimising residuals.

        Iterates over candidate inflection points from the second derivative.

        Parameters
        ----------
        Id : ndarray
            sqrt(|Id|) values.
        V : ndarray
            Corresponding gate voltages.

        Returns
        -------
        ndarray
            Best-fit coefficients [slope, intercept].
        """
        _residuals = np.array([])
        _fits = np.array([0, 0])

        # splines needs to be ascending
        if V[2] < V[1]:
            V = np.flip(V)
            Id = np.flip(Id)

        self.quadrant()

        if self.quad == 'I':  # top right

            Id = np.flip(Id)
            V = np.flip(-V)

        mx_d2 = self._find_peak(Id * 1000, V)  # *1000 improves numerical spline accuracy

        # sometimes for very small currents run into numerical issues
        if not mx_d2:
            mx_d2 = self._find_peak(Id * 1000, V, width=15)

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

        if self.quad == 'I':
            fit[0] *= -1

        return fit

    # linear curve-fitting
    @staticmethod
    def line_f(x, f0, f1):

        return f1 + f0 * x

    @staticmethod
    def _find_peak(I, V, width=15):
        """
        Finds the saturation-regime transition point via the second derivative of a spline.

        Parameters
        ----------
        I : ndarray
            Drain current values (sqrt-scaled).
        V : ndarray
            Gate voltage values.
        width : int, optional
            Minimum peak width passed to find_peaks.

        Returns
        -------
        list of int
            Indices in the original V array of candidate transition points.
        """

        # uses second derivative for transition point
        Id_spl = spi.UnivariateSpline(V, I, k=5, s=1e-7)
        V_spl = np.arange(V[0], V[-1], 0.005)
        d2 = np.gradient(np.gradient(Id_spl(V_spl)))

        peaks, _ = sps.find_peaks(d2, width=width)
        peaks = peaks[peaks > 5]  # edge errors

        # find splined index in original array
        mx_d2 = [np.searchsorted(V, V_spl[p]) for p in peaks]

        return mx_d2

    '''
    File loading and processing
    '''

    def loaddata(self):
        """
        3 Steps to loading a folder of data:
            1) generate filelist for only txt files
            2) determine if config exists (newer devices)
            3) for each file in the filelist, generate a transfer curve or output curve

        """

        for t in self.files:
            print(t)

            self.get_metadata(t)

            if 'transfer' in t.parts[-1]:
                self.transfer_curve(t)

            elif 'output' in t.parts[-1]:
                self.output_curve(t)

        self.all_outputs()
        self.all_transfers()

        self.num_transfers = len(self.transfers.columns)
        self.num_outputs = len(self.outputs.columns)

        if self.make_config:  # no proper config file found
            self.update_config()

        # can manually use options to overwrite the config file
        if 'overwrite' in self.options:
            if self.options['overwrite']:
                self.update_config()
        return

    def filelist(self):
        """
        Generates list of files to process and config file
        """

        filelist = os.listdir(self.folder)
        files = [pathlib.Path(os.path.join(self.folder, name))
                 for name in filelist if name[-3:] == 'txt']

        # find config file
        config = [pathlib.Path(os.path.join(self.folder, name))
                  for name in filelist if name[-4:] == '.cfg']

        if config:

            for f in files:

                if 'config' in str(f):
                    files.remove(f)

            self.config = config

        else:

            print('No config file found!')
            # path = pathlib.Path('\\'.join(files[0].parts[:-1]))
            path = pathlib.Path(*files[0].parts[:-1])
            self.config = make_config(path)
            self.make_config = True

        self.files = files

        return

    def get_metadata(self, fl):
        """
        Extracts Vds, Vg, and optional device dimensions from a data file header.

        Parameters
        ----------
        fl : Path
            Path to the data file.
        """

        # search params in first file in this folder for missing params
        h = open(fl)
        for line in h:
            if 'V_DS = ' in line or 'V_DS =' in line:
                self.Vd = float(line.split()[-1])
            if 'V_G = ' in line or 'V_G =' in line:
                self.Vg = float(line.split()[-1])

            # if no config file found, populate based on the raw data
            if 'Width/um' in line and (self.make_config or self.options['overwrite']):
                self.W = float(line.split()[-1])
            if 'Length/um' in line and (self.make_config or self.options['overwrite']):
                self.L = float(line.split()[-1])

        h.close()

        return

    def output_curve(self, path):
        """
        Loads a single Id-Vd output curve file into self.output.

        Parameters
        ----------
        path : Path
            Path to the output curve text file.
        """

        V = self.Vg

        op = pd.read_csv(path, delimiter='\t', engine='python')
        # Remove junk rows
        _junk = pd.to_numeric(op['V_DS'], errors='coerce')
        _junk = _junk.notnull()
        op = op.loc[_junk]
        op = op.set_index('V_DS')
        op = op.set_index(pd.to_numeric(op.index.values))

        mx, reverse = self._reverse(op.index.values, transfer=False)
        # idx = op.index.values[mx]

        self.Vg_array.append(V)
        Vfwd = str(V) + '_fwd'
        self.output[Vfwd] = op.iloc[:mx]
        self.output_raw[Vfwd] = op.iloc[:mx]
        self.output[Vfwd] = self.output[Vfwd].drop(labels=['I_DS Error (A)',
                                                           'I_G (A)',
                                                           'I_G Error (A)'], axis=1)
        if reverse:
            Vbwd = str(V) + '_bwd'
            self.output[Vbwd] = op.iloc[mx:]
            self.output_raw[Vbwd] = op.iloc[mx:]
            self.output[Vbwd] = self.output[Vbwd].drop(labels=['I_DS Error (A)',
                                                               'I_G (A)',
                                                               'I_G Error (A)'], axis=1)

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
        """
        Loads a single Id-Vg transfer curve file into self.transfer.

        Parameters
        ----------
        path : Path
            Path to the transfer curve text file.
        """
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
        self.transfer[transfer_Vd] = self.transfer[transfer_Vd].drop(labels=['I_DS Error (A)',
                                                                             'I_G (A)',
                                                                             'I_G Error (A)'],
                                                                     axis=1)

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

            mx, reverse = self._reverse(idx, transfer=True)
            nm = tf + '_01'
            df = pd.Series(data=transfer[:mx], index=idx[:mx])
            # df.sort_index(inplace=True)
            self.transfers[nm] = df

            if reverse:
                nm = tf + '_02'
                df = pd.Series(data=transfer[mx:], index=idx[mx:])
                # df.sort_index(inplace=True)
                self.transfers[nm] = df

        if 'Average' in self.options and self.options['Average']:
            self.transfers = pd.DataFrame(self.transfers.mean(1), columns=['transfer_avg'])

        # if there's an "inversion" at the end, finds that point
        if self.options['V_low'] is True:

            for e in self.transfers:

                df = self.transfers[e].copy()
                df.sort_index(inplace=True)

                vdx = df.index.values
                idx = df.values

                for x in np.arange(len(vdx) - 1):

                    if (idx[x + 1] - idx[x]) > 0:
                        break

                cut = x

                self.transfers[e] = self.transfers.iloc[cut:][e]

        return

    def update_config(self):

        config = configparser.ConfigParser()
        config.read(self.config)

        # Update with parameters read in earlier in loaddata()
        config['Dimensions']['Width (um)'] = str(self.W)
        config['Dimensions']['Length (um)'] = str(self.L)
        config['Transfer']['Vds (V)'] = str(self.Vd)

        config['Output'] = {'Preread (ms)': 500.0,
                            'First Bias (ms)': 200.0}
        config['Output']['Output Vgs'] = str(len(self.Vg_array))
        for v in range(len(self.Vg_array)):
            config['Output']['Vgs (V) ' + str(v)] = str(self.Vg_array[v])

        # overwrite the file
        try:
            with open(self.config, 'w') as configfile:

                config.write(configfile)
        except:

            with open(self.config[0], 'w') as configfile:

                config.write(configfile)

        return

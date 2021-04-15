# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:23:47 2019

@author: Raj
"""

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.integrate import trapz

'''
To add:
    Data as function of time instead of vs voltage
    Add plotting functions/labels
    Show integrated current
'''


class cv:

    def __init__(self, path):

        self.path = path

        cv = pd.read_csv(self.path, sep='\t')

        self.v = cv['WE(1).Potential (V)']
        self.i = cv['WE(1).Current (A)']
        self.t = cv['Time (s)']
        self.t -= self.t[0]

        self.slice_cv()

        return

    def slice_cv(self):

        self.df_time = pd.DataFrame()
        self.df_volt = pd.DataFrame()

        peaks = sps.find_peaks(self.v)[0]

        self.cycles = len(peaks)

        period = np.abs(peaks[1] - peaks[0])

        self.period = period
        vx = self.v[:period]
        tx = self.t[:period]

        for i in np.arange(self.cycles):
            self.df_volt[i] = self.i[period * i:period * (i + 1)].values
            self.df_time[i] = self.i[period * i:period * (i + 1)].values

        self.df_volt = self.df_volt.set_index(vx)
        self.df_time = self.df_time.set_index(tx)

        return

    def int_current(self):

        cvc = np.array([])
        p = int(self.period / 2)

        for c in self.df_time.columns:
            cvc = np.append(cvc, [trapz(self.df_time[c].iloc[:p], self.t[:p]),
                                  trapz(self.df_time[c].iloc[p:], self.t[:p])])

        cvc = np.reshape(cvc, [self.cycles, 2])

        df = pd.DataFrame()

        for i, x in zip(cvc, range(self.cycles)):
            df[x] = i

        df = df.rename({0: '+', 1: '-'}, axis='index')

        self.current_vs_cycle = df

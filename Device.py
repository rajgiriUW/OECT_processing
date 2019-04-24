# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:16:43 2019

@author: Raj
"""

import numpy as np
from scipy.optimize import curve_fit as cf

import oect_plot
import oect_load

import pickle

class OECTDevice:
    '''
    Class containing the processed pixels for a single device
    
    This simplifies comparing and plotting various uC* datasets together
    
    '''
    def __init__(self, path='', pixels={}, params={}, 
                 options={'V_low': False, 'retrace_only': False, 'verbose': False}):
    
        self.path = path
        
        if not path:
            
            from PyQt5 import QtWidgets
    
            app = QtWidgets.QApplication([])
            self.path = QtWidgets.QFileDialog.getExistingDirectory(caption='Select folder of data')
            print('Loading', self.path)
            app.closeAllWindows()
            app.exit()

        self.params = {}
        for m in params:
            self.params[m] = params[m]
        
        self.options = {}
        for o in options:
            self.options[o] = options[o]

        # if device has not been processed
        if not any(pixels): 
    
            pixels, pm = OECT_loading.uC_scale(self.path, 
                                                   V_low=options['V_low'],
                                                   retrace_only=options['retrace_only'],
                                                   verbose=options['verbose'])

            for m in pm:
                self.params[m] = pm[m]
    
        self.pixels = pixels

        # extract a subset as direct attributes
        self.L = self.params['L']
        self.WdL = self.params['WdL']
        self.W = self.params['W']
        self.d = self.params['d']
        self.Vg_Vt = self.params['Vg_Vt']
        self.Vt = self.params['Vt']
        self.uC = self.params['uC']
        self.uC_0 = self.params['uC_0']
        self.gms = self.params['gms']
        
        self.pix_paths = []
        
        for p in self.pixels:
            self.pix_paths.append(self.pixels[p].folder)
        
        return
    
    def get_params(self):
        '''
        Generates the parameters from the pixel data
        '''
        Wd_L = np.array([])
        W = np.array([])
        Vg_Vt = np.array([])  # threshold offset
        Vt = np.array([])
        gms = np.array([])
    
        # assumes Length and thickness are fixed
        params = {}
    
        for pixel in self.pixels:
            
            if not self.pixels[pixel].gms.empty:
                
                ix = len(self.pixels[pixel].VgVts)
                Vt = np.append(Vt, self.pixels[pixel].Vts)
                Vg_Vt = np.append(Vg_Vt, self.pixels[pixel].VgVts)
                gms = np.append(gms, self.pixels[pixel].gm_peaks['peak gm (S)'].values)
                W = np.append(W, self.pixels[pixel].W)
                
                # appends WdL as many times as there are transfer curves
                for i in range(len(self.pixels[pixel].VgVts)):
                    Wd_L = np.append(Wd_L, self.pixels[pixel].WdL)
                # remove the trace ()
                if self.options['retrace_only'] and len(self.pixels[pixel].VgVts) > 1:
                    Vt = np.delete(Vt, -ix)
                    Vg_Vt = np.delete(Vg_Vt, -ix)
                    gms = np.delete(gms, -ix)
                    Wd_L = np.delete(Wd_L, -ix)
        
                params['L'] = self.pixels[pixel].L
                params['d'] = self.pixels[pixel].d
                
        
        # fit functions
        def line_f(x, a, b):
    
            return a + b * x
    
        def line_0(x, b):
            'no y-offset --> better log-log fits'
            return b * x
    
        # * 1e2 to get into right mobility units (cm)
        uC_0, _ = cf(line_0, Wd_L * Vg_Vt, gms)
        uC, _ = cf(line_f, Wd_L * Vg_Vt, gms)
    
        # Create an OECT and add arrays 
        params['WdL'] = Wd_L
        params['W'] = W
        params['Vg_Vt'] = Vg_Vt
        params['Vt'] = Vt
        params['uC'] = uC
        params['uC_0'] = uC_0
        params['gms'] = gms
        
        self.params = params
        
        self.L = self.params['L']
        self.WdL = self.params['WdL']
        self.W = self.params['W']
        self.d = self.params['d']
        self.Vg_Vt = self.params['Vg_Vt']
        self.Vt = self.params['Vt']
        self.uC = self.params['uC']
        self.uC_0 = self.params['uC_0']
        self.gms = self.params['gms']
        
        return
    
    def plot_uc(self, save=False):
        
        fig = oect_plot.plot_uC(self.params, savefig=save)
        
        return
    
def save(dv, append=''):
    
    with open(dv.path+ r'\uC_data_'+append+'.pkl', 'wb') as output:
        
        pickle.dump(dv, output, pickle.HIGHEST_PROTOCOL)
    
    return    
        

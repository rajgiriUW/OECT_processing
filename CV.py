# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:23:47 2019

@author: Raj
"""

import numpy as np
import pandas as pd

class cv:
    
        def __init__(self, path):
            
            self.path = path
            
            cv = pd.read_csv(self.path, sep='\t')
            
            self.v = cv['WE(1).Potential (V)']
            self.i = cv['WE(1).Current (A)']
            self.t = cv['Time (s)']
            self.t -= self.t[0]
            
            return
        
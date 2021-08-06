import os
import sys
import pytest
import configparser
import numpy as np
import pathlib

sys.path.insert(0, '..')

print(os.getcwd())
if 'tests' in os.getcwd():
    os.chdir('..')
sys.path.append('oect/')

import oect_processing
from oect_processing.specechem import uvvis, read_files, uvvis_h5, uvvis_plot
from oect_processing.nonoect_utils import cv

# most values are hardcoded - be careful if modifying cfg/txt files
# some tests use different subfolders to avoid conflicts

class TestUVVis:
    
    # set_params
    ############################################################################
    
    # test read_files
    def test_read_files(self):
        
        pass
#use this file to manage uvvis function inputs.
#these values are imported to uvvis_testing.py and generate_uvvis_expected_values.py
#please refer to the readme in the containing folder

#imports
import pandas as pd
import numpy as np

#class variables
data_folder = "../test_specechem_with_dedoping"
wavelength = 500
potential = 0.6
droptimes = []


#time_dep_spectra
time_dep_smooth = None
time_dep_round_wl = 2
time_dep_droptimes = None

#_single_time_spectra
single_time_smooth = 3
single_time_digits = None

#spec_echem_voltage
spec_echem_time = 0
spec_echem_smooth = 3
spec_echem_digits = None

#time_index
##these are not function inputs but are needed because time_index works 
##on uvvis.spectra_vs_time. these dataframes works as dummy dataframes to make up
##an arbitrary spectra_vs_time.
random_df_1 = pd.DataFrame([np.arange(5), np.arange(3)])
random_df_2 = pd.DataFrame([np.arange(4), np.arange(6)])
random_df_3 = pd.DataFrame([np.arange(4), np.arange(5)])

#single_wl_time
single_wl_potential = 0.9
single_wl_wavelength = 800
single_wl_smooth = 3

#abs_vs_voltage
abs_v_wl_wavelength = 800
abs_v_time = 0

#banded_fits
fits_wl_start = 700
fits_wl_stop = 900
fits_voltage = 1
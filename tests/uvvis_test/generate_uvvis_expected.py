# this file generates expected values with which uvvis_test.py can compare.
# run this when you are sure the functions work as expected, or add print
# statements to check that they work as expected
# please refer to the readme in the containing folder.

# as of august 27 2021, all functions except for banded_fits(fittype="biexp") work as expected.

import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl

sys.path.insert(0, '../..')
import oect_processing
from oect_processing.specechem import uvvis, read_files
from tests.uvvis_test import uvvis_inputs

expected_values_folder = "uvvis_expected_values"
if not os.path.isdir(expected_values_folder):
    os.makedirs(expected_values_folder)

steps, specs, volts, dedopesteps, dedopespecs = \
    read_files.read_files(os.path.join("..", uvvis_inputs.data_folder))

data = uvvis.UVVis(steps, specs, volts)
dedata = uvvis.UVVis(dedopesteps, dedopespecs, volts)

# single_time_spectra
df_data = data._single_time_spectra(spectra_path=specs[0], \
                                    smooth=uvvis_inputs.single_time_smooth, digits=uvvis_inputs.single_time_digits)
df_data.to_pickle(expected_values_folder + "/expected_data_single_time_spectra.pkl")

df_dedata = dedata._single_time_spectra(spectra_path=dedopespecs[0], \
                                        smooth=uvvis_inputs.single_time_smooth, digits=uvvis_inputs.single_time_digits)
df_data.to_pickle(expected_values_folder + "/expected_dedata_single_time_spectra.pkl")

# TIME_INDEX
df_dict = {0: uvvis_inputs.random_df_1, 1: uvvis_inputs.random_df_2, 2: uvvis_inputs.random_df_3}
data.spectra_vs_time = df_dict
data.time_index()
np.save(expected_values_folder + "/expected_time_index.npy", data.tx)

# TIME_DEP_SPECTRA
data.time_dep_spectra(specfiles=specs, smooth=uvvis_inputs.time_dep_smooth, \
                      round_wl=uvvis_inputs.time_dep_round_wl, droptimes=uvvis_inputs.droptimes)
np.save(expected_values_folder + "/expected_data_spectra_vs_time.npy", data.spectra_vs_time)

dedata.time_dep_spectra(specfiles=dedopespecs, smooth=uvvis_inputs.time_dep_smooth, \
                        round_wl=uvvis_inputs.time_dep_round_wl, droptimes=uvvis_inputs.droptimes)
np.save(expected_values_folder + "/expected_dedata_spectra_vs_time.npy", dedata.spectra_vs_time)

# SINGLE_WL_TIME
data.single_wl_time(wavelength=uvvis_inputs.single_wl_wavelength, \
                    potential=uvvis_inputs.single_wl_potential, smooth=uvvis_inputs.single_wl_smooth)
np.save(expected_values_folder + "/expected_data_time_spectra.npy", data.time_spectra)
np.save(expected_values_folder + "/expected_data_time_spectra_norm.npy", data.time_spectra_norm)
np.save(expected_values_folder + "/expected_data_time_spectra_sm.npy", data.time_spectra_sm)
np.save(expected_values_folder + "/expected_data_time_spectra_norm_sm.npy", data.time_spectra_norm_sm)

dedata.single_wl_time(wavelength=uvvis_inputs.single_wl_wavelength, \
                      potential=uvvis_inputs.single_wl_potential, smooth=uvvis_inputs.single_time_smooth)
np.save(expected_values_folder + "/expected_dedata_time_spectra.npy", dedata.time_spectra)
np.save(expected_values_folder + "/expected_dedata_time_spectra_norm.npy", dedata.time_spectra_norm)
np.save(expected_values_folder + "/expected_dedata_time_spectra_sm.npy", dedata.time_spectra_sm)
np.save(expected_values_folder + "/expected_dedata_time_spectra_norm_sm.npy", dedata.time_spectra_norm_sm)

# SPEC_ECHEM_VOLTAGE
data.spec_echem_voltage(time=uvvis_inputs.spec_echem_time, \
                        smooth=uvvis_inputs.spec_echem_smooth, digits=uvvis_inputs.spec_echem_smooth)
data.spectra.to_pickle(expected_values_folder + "/expected_data_spectra.pkl")
data.spectra_sm.to_pickle(expected_values_folder + "/expected_data_spectra_sm.pkl")

dedata.spec_echem_voltage(time=uvvis_inputs.spec_echem_time, \
                          smooth=uvvis_inputs.spec_echem_smooth, digits=uvvis_inputs.spec_echem_smooth)
dedata.spectra.to_pickle(expected_values_folder + "/expected_dedata_spectra.pkl")
dedata.spectra_sm.to_pickle(expected_values_folder + "/expected_dedata_spectra_sm.pkl")

# CURRENT_VS_TIME
data.current_vs_time(stepfiles=steps)
data.current.to_pickle(expected_values_folder + "/expected_data_current.pkl")
data.charge.to_pickle(expected_values_folder + "/expected_data_charge.pkl")

dedata.current_vs_time(stepfiles=dedopesteps)
dedata.current.to_pickle(expected_values_folder + "/expected_dedata_current.pkl")
dedata.charge.to_pickle(expected_values_folder + "/expected_dedata_charge.pkl")

# ABS_VS_VOLTAGE
data.abs_vs_voltage()
data.vt.to_pickle(expected_values_folder + "/expected_data_abs_vs_volt.pkl")

dedata.abs_vs_voltage()
dedata.vt.to_pickle(expected_values_folder + "/expected_dedata_abs_vs_volt.pkl")

# BANDED_FITS
data.banded_fits(fittype="exp")
np.save(expected_values_folder + "/expected_data_exp_banded_fits.npy", data.fits)

###data.banded_fits(fittype="biexp")
###np.save(expected_values_folder + "/expected_data_biexp_banded_fits.npy", data.fits)

data.banded_fits(fittype="stretched")
np.save(expected_values_folder + "/expected_data_stretched_banded_fits.npy", data.fits)

dedata.banded_fits(fittype="exp")
np.save(expected_values_folder + "/expected_dedata_exp_banded_fits.npy", dedata.fits)

###dedata.banded_fits(fittype="biexp")
###np.save(expected_values_folder + "/expected_data_biexp_banded_fits.npy", dedata.fits)

dedata.banded_fits(fittype="stretched")
np.save(expected_values_folder + "/expected_dedata_stretched_banded_fits.npy", dedata.fits)

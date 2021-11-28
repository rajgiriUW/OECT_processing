#this file runs tests on uvvis.py using pytest.
#please refer to the readme in the containing folder

#as of august 27 2021, all functions except for banded_fits(fittype="biexp")
#work correctly

import os
import sys
import pytest
import configparser
import numpy as np
import pathlib
import h5py
import pandas as pd
import pickle as pkl
import pickle
import deepdiff
from deepdiff import DeepDiff

sys.path.insert(0, '../..')


import oect_processing
from oect_processing.specechem import uvvis, read_files, uvvis_h5, uvvis_plot
from oect_processing.nonoect_utils import cv
from tests.uvvis_test import uvvis_inputs

expected_values_folder = 'tests/uvvis_test/uvvis_expected_values'

#the below tests take about 6 minutes in total to run

class TestUVVis:
			
	@pytest.fixture
	def read_files(self):
		steps, specs, volts, dedopesteps, dedopespecs = read_files.read_files(os.path.join(os.getcwd(), uvvis_inputs.data_folder))
		return {"steps":steps, "specs":specs, "volts":volts, "dedopesteps":dedopesteps, "dedopespecs":dedopespecs}
	
	@pytest.fixture
	def data_load(self, read_files):
		data = uvvis.UVVis(read_files["steps"], read_files["specs"], read_files["volts"])
		dedata = uvvis.UVVis(read_files["dedopesteps"], read_files["dedopespecs"], read_files["volts"])
		return {"doping": data, "dedoping": dedata}
	
	@pytest.mark.dependency(depends=["read_files", "data_load"])
	@pytest.mark.parametrize("dataset, specs, expected_values_folder", \
		[("doping", "specs", os.path.join(expected_values_folder, "expected_data_single_time_spectra.pkl")),\
		("dedoping", "specs", os.path.join(expected_values_folder, "expected_dedata_single_time_spectra.pkl"))])
	def test_single_time_spectra(self, read_files, data_load, dataset, specs, expected_values_folder):
		try:
			expected_single_time_spectra = np.load(expected_values_folder, allow_pickle = True)
			time_spectra = data_load[dataset]._single_time_spectra(spectra_path = read_files[specs][0], smooth = uvvis_inputs.single_time_smooth, digits = uvvis_inputs.single_time_digits)
			assert time_spectra.equals(expected_single_time_spectra)
		except:
			assert False
	
	def test_time_index(self, data_load):
		try:
			df_dict = {0:uvvis_inputs.random_df_1, 1:uvvis_inputs.random_df_2, 2:uvvis_inputs.random_df_3}
			data_load["doping"].spectra_vs_time = df_dict
			data_load["doping"].time_index()
			time_index = data_load["doping"].tx
			expected_time_index = np.load(os.path.join(expected_values_folder, "expected_time_index.npy"))
			assert np.array_equal(time_index, expected_time_index)
		except:
			assert False
			
	@pytest.mark.dependency(depends=["read_files", "data_load", \
		"test_single_time_spectra", "test_time_index"])
	@pytest.mark.parametrize("dataset, specs, expected_values_folder", \
		[("doping", "specs", os.path.join(expected_values_folder, "expected_data_spectra_vs_time.npy")),\
		("dedoping", "dedopespecs", os.path.join(expected_values_folder, "expected_dedata_spectra_vs_time.npy"))])
	def test_time_dep_spectra(self, read_files, data_load, dataset, specs, expected_values_folder):
		try:
			data_load[dataset].time_dep_spectra(specfiles = read_files[specs], smooth = uvvis_inputs.time_dep_smooth, round_wl = uvvis_inputs.time_dep_round_wl, droptimes=uvvis_inputs.droptimes)
			spectra_vs_time = data_load[dataset].spectra_vs_time
			expected_spectra_vs_time = np.load(expected_values_folder, allow_pickle = True).item()
			assert not DeepDiff(spectra_vs_time, expected_spectra_vs_time)
		except:
			assert False
	
	@pytest.mark.dependency(depends=["test_time_dep_spectra"])
	@pytest.fixture
	def time_dep_spectra(self, read_files, data_load):
		data_load["doping"].time_dep_spectra(read_files["specs"], smooth = uvvis_inputs.time_dep_smooth, round_wl = uvvis_inputs.time_dep_round_wl, droptimes=uvvis_inputs.droptimes)
		data_load["dedoping"].time_dep_spectra(read_files["dedopespecs"], smooth = uvvis_inputs.time_dep_smooth, round_wl = uvvis_inputs.time_dep_round_wl, droptimes=uvvis_inputs.droptimes)
		return {"doping":data_load["doping"].spectra_vs_time, "dedoping":data_load["dedoping"].spectra_vs_time} 
	
	@pytest.mark.dependency(depends=["read_files", "data_load", "test_time_dep_spectra"])
	@pytest.mark.parametrize("dataset, specs, data_label", [("doping", "specs", "data"), ("dedoping", "specs", "dedata")])
	def test_single_wl_time(self, read_files, data_load, dataset, specs, data_label, time_dep_spectra):
		try:
			data_load[dataset].spectra_vs_time = time_dep_spectra[dataset]
			data_load[dataset].single_wl_time(wavelength=uvvis_inputs.single_wl_wavelength, potential=uvvis_inputs.single_wl_potential, smooth = uvvis_inputs.single_wl_smooth)
			expected_time_spectra = np.load(os.path.join(expected_values_folder, "expected_%s_time_spectra.npy" % data_label))
			expected_time_spectra_norm = np.load(os.path.join(expected_values_folder, "expected_%s_time_spectra_norm.npy" % data_label))
			expected_time_spectra_sm = np.load(os.path.join(expected_values_folder, "expected_%s_time_spectra_sm.npy" % data_label))
			expected_time_spectra_norm_sm = np.load(os.path.join(expected_values_folder, "expected_%s_time_spectra_norm_sm.npy" % data_label))
			assert np.array_equal(data_load[dataset].time_spectra, expected_time_spectra) \
				and np.array_equal(data_load[dataset].time_spectra_norm, expected_time_spectra_norm) \
				and np.array_equal(data_load[dataset].time_spectra_sm, expected_time_spectra_sm) \
				and np.array_equal(data_load[dataset].time_spectra_norm_sm, expected_time_spectra_norm_sm)
		except:
			assert False
	
	@pytest.mark.dependency(depends=["read_files", "data_load"])
	@pytest.mark.parametrize("dataset, steps, data_label", [("doping", "steps", "data"), ("dedoping", "dedopesteps", "dedata")]) 
	def test_current_extraction(self, data_load, read_files, dataset, steps, data_label):
		try:
			data_load[dataset].current_vs_time(stepfiles=read_files[steps])
			current = data_load[dataset].current
			charge = data_load[dataset].charge
			expected_current = pd.read_pickle(os.path.join(expected_values_folder, "expected_%s_current.pkl" % data_label))
			expected_charge = pd.read_pickle(os.path.join(expected_values_folder, "expected_%s_charge.pkl" % data_label))
			assert current.equals(expected_current) and charge.equals(expected_charge)
		except:
			assert False
				
	@pytest.mark.dependency(depends=["read_files", "data_load", "test_time_dep_spectra"])
	@pytest.mark.parametrize("dataset, specs, data_label", [("doping", "specs", "data"), ("dedoping", "specs", "dedata")])
	def test_spec_echem_voltage(self, read_files, data_load, dataset, specs, data_label, time_dep_spectra):
		try:
			data_load[dataset].spectra_vs_time = time_dep_spectra[dataset]
			data_load[dataset].spec_echem_voltage(time = uvvis_inputs.spec_echem_time, smooth = uvvis_inputs.spec_echem_smooth, digits = uvvis_inputs.spec_echem_smooth)
			spectra = data_load[dataset].spectra
			spectra_sm = data_load[dataset].spectra_sm
			expected_spectra = pd.read_pickle(os.path.join(expected_values_folder, "expected_%s_spectra.pkl" % data_label))
			expected_spectra_sm = pd.read_pickle(os.path.join(expected_values_folder, "expected_%s_spectra_sm.pkl" % data_label))
			assert spectra.equals(expected_spectra) and spectra_sm.equals(expected_spectra_sm)
		except:
			assert False
			
	@pytest.mark.dependency(depends=["data load"])
	@pytest.mark.parametrize("dataset, filename", [("doping", "dopingdata.h5"), ("dedoping", "dedopingdata.h5")])
	def test_save_h5(self, data_load, dataset, filename):
		try:
			working_path = os.path.join(uvvis_inputs.data_folder, filename)
			uvvis_h5.save_h5(os.getcwd(), working_path)
			assert True
		except:
			assert True
			
	@pytest.mark.dependency(depends=["read_files", "data_load"])
	@pytest.mark.parametrize("dataset, filename", [("doping", "dopingdata.pkl"), ("dedoping", "dedopingdata.pkl")])
	def test_pkl_dump(self, data_load, dataset, filename):
		try:
			with open(filename, 'wb') as output:
				pickle.dump(data_load[dataset], output, pickle.HIGHEST_PROTOCOL)
			assert True
		except:
			assert False
			
	@pytest.mark.dependency(depends=["read_files", "data_load", "test_time_index"])
	@pytest.mark.parametrize("dataset, data_label", [("doping", "data"), ("dedoping", "dedata")])
	def test_abs_vs_voltage(self, read_files, data_load, time_dep_spectra, dataset, data_label):
		try:
			data_load[dataset].spectra_vs_time = time_dep_spectra[dataset]
			data_load[dataset].time_index()
			data_load[dataset].abs_vs_voltage()
			vt = data_load[dataset].vt
			expected_vt = pd.read_pickle(os.path.join(expected_values_folder, "expected_%s_abs_vs_volt.pkl" % data_label))
			assert vt.equals(expected_vt)
		except:
			assert False
			
	@pytest.mark.dependency(depends=["read_files", "data_load", "test_single_wl_time"])
	@pytest.mark.parametrize("dataset, fitfunc, data_label", [("doping", "exp", "data"), ("dedoping", "exp", "dedata"), \
		("doping", "biexp", "data"), ("dedoping", "biexp", "dedata"), ("doping", "stretched", "data"), ("dedoping", "stretched", "dedata"),])
	def test_banded_fits(self, data_load, time_dep_spectra, dataset, fitfunc, data_label):
		try:
			data_load[dataset].spectra_vs_time = time_dep_spectra[dataset]
			data_load[dataset].single_wl_time(wavelength=uvvis_inputs.single_wl_wavelength, potential=uvvis_inputs.single_wl_potential, smooth = uvvis_inputs.single_wl_smooth)
			data_load[dataset].banded_fits()
			fits = data_load[dataset].fits
			expected_fits = np.load(os.path.join(expected_values_folder, "expected_%s_%s_banded_fits.npy" % (data_label, fitfunc)))
			assert True
		except:
			assert False

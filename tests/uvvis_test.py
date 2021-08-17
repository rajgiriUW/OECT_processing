import os
import sys
import pytest
import configparser
import numpy as np
import pathlib
import h5py

sys.path.insert(0, '..')


import oect_processing
from oect_processing.specechem import uvvis, read_files, uvvis_h5, uvvis_plot
from oect_processing.nonoect_utils import cv
import pickle
 
#change these values as needed
DATA_FOLDER = "test_specechem_with_dedoping"
WAVELENGTH = 500
POTENTIAL = 0.6
DROPTIMES = []

#the below tests take about 5 minutes in total to run
class TestUVVis:
	
	overwrite = False
	
	@pytest.fixture
	def read_files(self):
		steps, specs, volts, dedopesteps, dedopespecs = read_files.read_files(DATA_FOLDER)
		return {"steps":steps, "specs":specs, "volts":volts, "dedopesteps":dedopesteps, "dedopespecs":dedopespecs}
		
	@pytest.fixture
	@pytest.mark.dependency(depends=["read_files"])
	def data_load(self, read_files):
		if not self.overwrite:
			try:
				working_path = os.path.join(DATA_FOLDER, "dopingdata.pkl")
				with open(working_path, 'rb') as input:
					data = pickle.load(input)
				print('Loaded existing doping data')
				working_path = os.path.join(DATA_FOLDER, "dedopingdata.pkl")
				with open(working_path, 'rb') as input:
					dedata = pickle.load(input)
				print('Loaded existing dedoping data')
				print('If you REALLY want to re-process all the data, change overwrite to True. You should not need to do this.')
			except:
				self.overwrite = True
		data = uvvis.UVVis(read_files["steps"], read_files["specs"], read_files["volts"])
		dedata = uvvis.UVVis(read_files["dedopesteps"], read_files["dedopespecs"], read_files["volts"])
		return {"doping":data, "dedoping":dedata}

	@pytest.mark.dependency(depends=["read_files", "data_load"])
	@pytest.mark.parametrize("dataset, specs", [("doping", "specs"), ("dedoping", "specs")])
	def test_time_dep_spectra(self, read_files, data_load, dataset, specs):
		try:
			data_load[dataset].time_dep_spectra(read_files[specs], droptimes=DROPTIMES)
			assert True, dataset + " time_dependent_spectra success"
		except:
			assert False, dataset + " time_dependent_spectra fail"
	
	@pytest.mark.dependency(depends=["test_time_dep_spectra"])
	@pytest.fixture
	def time_dep_spectra(self, read_files, data_load):
		data_load["doping"].time_dep_spectra(read_files["specs"], droptimes=DROPTIMES)
		data_load["dedoping"].time_dep_spectra(read_files["dedopespecs"], droptimes=DROPTIMES)
		return {"doping":data_load["doping"].spectra_vs_time, "dedoping":data_load["dedoping"].spectra_vs_time} 
	
			
	@pytest.mark.dependency(depends=["read_files", "data_load", "test_time_dep_spectra"])
	@pytest.mark.parametrize("dataset, specs", [("doping", "specs"), ("dedoping", "specs")])
	def test_single_wl_time(self, read_files, data_load, dataset, specs, time_dep_spectra):
		try:
			data_load[dataset].spectra_vs_time = time_dep_spectra[dataset]
			data_load[dataset].single_wl_time(wavelength=WAVELENGTH, potential=POTENTIAL)
			assert True, dataset + " single_wl_time success"
		except:
			assert False, dataset + " single_wl_time fail"
	
	@pytest.mark.dependency(depends=["read_files", "data_load"])
	@pytest.mark.parametrize("dataset, steps", [("doping", "steps"), ("dedoping", "dedopesteps")]) 
	def test_current_extraction(self, data_load, read_files, dataset, steps):
		try:
			data_load[dataset].current_vs_time(stepfiles=read_files[steps])
			assert True, dataset + " current_extraction success"
		except:
			assert False, dataset + " current_extraction fail"
				
	@pytest.mark.dependency(depends=["read_files", "data_load", "test_time_dep_spectra"])
	@pytest.mark.parametrize("dataset, specs", [("doping", "specs"), ("dedoping", "specs")])
	def test_time_slice_spectra(self, read_files, data_load, dataset, specs, time_dep_spectra):
		try:
			data_load[dataset].spectra_vs_time = time_dep_spectra[dataset]
			data_load[dataset].spec_echem_voltage(time=0)
			assert True, dataset + " time_slice_spectra successful"
		except:
			assert False, dataset + " time_slice_spectra failed"
			
	@pytest.mark.dependency(depends=["data load"])
	@pytest.mark.parametrize("dataset, filename", [("doping", "dopingdata.h5"), ("dedoping", "dedopingdata.h5")])
	def test_save_h5(self, data_load, dataset, filename):
		try:
			working_path = os.path.join(DATA_FOLDER, filename)
			uvvis_h5.save_h5(dataset, working_path)
			assert True, "save " + dataset + " as h5 successful"
		except:
			assert True, "save " + dataset + " as h5 failed"
			
	@pytest.mark.dependency(depends=["read_files", "data_load"])
	@pytest.mark.parametrize("dataset, filename", [("doping", "dopingdata.h5"), ("dedoping", "dedopingdata.h5")])
	def test_pkl_dump(self, data_load, dataset, filename):
		try:
			working_path = os.path.join(DATA_FOLDER, filename)
			with open(working_path, 'wb') as output:
				pickle.dump(data_load[dataset], output, pickle.HIGHEST_PROTOCOL)
			assert True, dataset + " pkl_dump successful"
		except:
			assert False, dataset + " pkl_dump fail"
	
	@pytest.mark.dependency(depends=["read_files", "data_load"])
	@pytest.mark.parametrize("dataset, specs", [("doping", "specs"), ("dedoping", "specs")])
	def test_single_time_spectra(self, read_files, data_load, dataset, specs):
		try:
			data_load[dataset]._single_time_spectra(read_files[specs][0])
			assert True, dataset + " single_time_spectra success"
		except:
			assert False, dataset + " single_time_spectra fail"
			
	@pytest.mark.dependency(depends=["read_files", "data_load", "test_time_dep_spectra"])
	@pytest.mark.parametrize("dataset", ["doping", "dedoping"])
	def test_spec_echem_voltage(self, read_files, data_load, time_dep_spectra, dataset):
		try:
			data_load[dataset].spectra_vs_time = time_dep_spectra[dataset]
			data_load[dataset].spec_echem_voltage()
			assert True, dataset + " spec_echem_voltage success"
		except:
			assert False, dataset + " spec_echem_voltage fail"

	@pytest.mark.dependency(depends=["read_files", "data_load", "test_time_dep_spectra"])
	@pytest.mark.parametrize("dataset", ["doping", "dedoping"])
	def test_time_index(self, read_files, data_load, time_dep_spectra, dataset):
		try:
			data_load[dataset].spectra_vs_time = time_dep_spectra[dataset]
			data_load[dataset].time_index()
			assert True, dataset + " time index success"
		except:
			assert False, dataset + " time index fail"
	
	@pytest.mark.dependency(depends=["read_files", "data_load", "test_time_index"])
	@pytest.mark.parametrize("dataset", ["doping", "dedoping"])
	def test_abs_vs_voltage(self, read_files, data_load, time_dep_spectra, dataset):
		try:
			data_load[dataset].spectra_vs_time = time_dep_spectra[dataset]
			data_load[dataset].time_index()
			data_load[dataset].abs_vs_voltage()
			assert True, dataset + " abs_vs_voltage success"
		except:
			assert False, dataset + " abs_vs_voltage fail"
			
	@pytest.mark.dependency(depends=["read_files", "data_load"])
	@pytest.mark.parametrize("dataset", ["doping", "dedoping"])
	def banded_fits(self, data_load, time_dep_spectra, dataset):
		try:
			data_load[dataset].spectra_vs_time = time_dep_spectra[dataset]
			data_load[dataset].banded_fits()
			assert True, dataset + " abs_vs_voltage success"
		except:
			assert False, dataset + " abs_vs_voltage fail"	

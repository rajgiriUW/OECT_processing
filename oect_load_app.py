# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:04:47 2018

@author: Raj
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit as cf

import oect
import oect_plot
import matplotlib as plt
from collections import Counter

'''
load_uC : for loading the four pixels to generate a uC* plot
load_avg : for loading all the subfolders that are the 4 "averaging" pixels
	this is somewhat uncommon

Usage:
	
	>> pixels, uC_dv = oect_load.uC_scale(r'path_to_uC_scale', new_geom=False)
	
'''


def uC_scale(paths, average_devices=False, dimDict={}, thickness=40e-9, plot=[True, False], V_low=False,
			 retrace_only=False, verbose=True, options={}, pg_graphs=[None, None], dot_color='r', text_browser=None):
	'''
	paths: array
		contains subfolders to plot
	
	average_devices : boolean
		whether to average devices of same WdL

	dimDict: dict
		dictionary in format of {parentfolder1: {subfolder1: w1, l1}, {subfolder2: w2, l2}, parentfolder2...}
	  
	thickness : float
		approximate film thickness. Standard polymers (for Raj) are ~40 nm
		
	plot : list of bools, optional
		[0]: Plot the uC* data
		[1]: plot the individual plots
		Whether to plot or not. Not plotting is very fast!    

	retrace_only : bool, optional
		Whether to only do the retrace in case trace isn't saturating
		
	V_low : bool, optional
		Whether to find erroneous "turnover" points when devices break down

	verbose: bool, optional
		Print to display

	#the following two will be passed to plot_uC in oect_plot.py
	pg_graphs: array of PlotItem
		UI graphs on which to plot.
		pg_graphs[0] is linear plot
		pg_graphs[1] is log plot
	dot_color: QColor
		color to use when plotting on pg_graphs

	text_browser: QTextBrowser
		UI text browser dispaying information

	Returns
	-------
	pixels : dict of OECT
		Contains the various OECT class devices
		
	uC_dv : dict containing
		Wd_L : ndarray
			coefficient for plotting on x-axis
		
		gms : ndarray
			average transconductance for plotting on y-axis
		
		Vg_Vt : ndarray
			threshold voltage shifts for correcting uC* fit
	
	'''
	pixkeys = [f + '_uC' for f in paths]

	# removes random files instead of the sub-folders
	for p in paths:
		if not os.path.isdir(p):
			paths.remove(p)

	pixels = {}
	opts = {'V_low': V_low}
	if any(options):
		for o in options:
			opts[o] = options[o]

	# loads all the folders
	if type(plot) == bool or len(plot) == 1:
		plot = [plot, plot]

	for p, f in zip(paths, pixkeys):

		if os.listdir(p):

			if verbose:
				print(p)
				if text_browser:
					text_browser.append(p)
			dv = loadOECT(p, dimDict, {'d': thickness}, gm_plot=plot, plot=plot[1],
						  options=opts, verbose=verbose, text_browser=text_browser)
			pixels[f] = dv
		else:

			pixkeys.remove(f)

	# do uC* graphs, need gm vs W*d/L
	Wd_L = np.array([])
	W = np.array([])
	Vg_Vt = np.array([])  # threshold offset
	Vt = np.array([])
	gms = np.array([])

	# assumes Length and thickness are fixed
	uC_dv = {}
	
	if average_devices:
		pixels = average_same_widths(pixels)

	for pixel in pixels:
		if not pixels[pixel].gms.empty:
			
			ix = len(pixels[pixel].VgVts)
			Vt = np.append(Vt, pixels[pixel].Vts)
			Vg_Vt = np.append(Vg_Vt, pixels[pixel].VgVts)
			gms = np.append(gms, pixels[pixel].peak_gm)
			W = np.append(W, pixels[pixel].W)

			# appends WdL as many times as there are transfer curves
			for i in range(len(pixels[pixel].VgVts)):
				Wd_L = np.append(Wd_L, pixels[pixel].WdL)
			# remove the trace ()
			if retrace_only and len(pixels[pixel].VgVts) > 1:
				Vt = np.delete(Vt, -ix)
				Vg_Vt = np.delete(Vg_Vt, -ix)
				gms = np.delete(gms, -ix)
				Wd_L = np.delete(Wd_L, -ix)

			uC_dv['L'] = pixels[pixel].L
			uC_dv['d'] = pixels[pixel].d

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
	uC_dv['WdL'] = Wd_L
	uC_dv['W'] = W
	uC_dv['Vg_Vt'] = Vg_Vt
	uC_dv['Vt'] = Vt
	uC_dv['uC'] = uC
	uC_dv['uC_0'] = uC_0
	uC_dv['gms'] = gms
	uC_dv['folder'] = os.path.dirname(paths[0])

	if plot[0]:
		fig = oect_plot.plot_uC(uC_dv, pg_graphs, dot_color=dot_color)

		if verbose:
			if text_browser:
				text_browser.append('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
			print('uC* = ', str(uC_0 * 1e-2), ' F/cm*V*s')

	if verbose:
		print('Vt = ', uC_dv['Vt'])
		if text_browser:
			text_browser.append('Vt = ' + str(uC_dv['Vt']))

	return pixels, uC_dv



def loadOECT(path, dimDict, params=None, gm_plot=True, plot=True, options={}, verbose=True, text_browser=None):
	"""
	Wrapper function for processing OECT data

	params = {W: , L: , d: } for W, L, d of device

	USAGE:
		device1 = loadOECT(folder_name)
	"""

	if not path:
		path = file_open(caption='Select device subfolder')

	device = oect.OECT(path, dimDict, params, options)
	device.calc_gms()
	device.thresh()

	scaling = device.WdL  # W *d / L

	if verbose:
		for key in device.gms:
			print(key, ': {:.2f}'.format(np.max(device.gms[key].values * 1e-2) / scaling), 'S/cm scaled')
			print(key, ': {:.2f}'.format(np.max(device.gms[key].values * 1000)), 'mS max')
			if (text_browser):
				text_browser.append(key + str(': {:.2f}'.format(np.max(device.gms[key].values * 1e-2) / scaling)) + 'S/cm scaled')
				text_browser.append(key + str(': {:.2f}'.format(np.max(device.gms[key].values * 1000))) + 'mS max')



	if plot:
		fig = oect_plot.plot_transfers_gm(device, gm_plot=gm_plot, leakage=True)
		fig.savefig(path + r'\transfer_leakage.tif', format='tiff')
		fig = oect_plot.plot_transfers_gm(device, gm_plot=gm_plot, leakage=False)
		fig.savefig(path + r'\transfer.tif', format='tiff')
		fig = oect_plot.plot_outputs(device, leakage=True)
		fig.savefig(path + r'\output_leakage.tif', format='tiff')
		fig = oect_plot.plot_outputs(device, leakage=False)
		fig.savefig(path + r'\output.tif', format='tiff')
	return device


def file_open(caption='Select folder'):
	'''
	File dialog if path not given in load commands
	:param
		caption : str

	:return:
		path : str
	'''

	from PyQt5 import QtWidgets

	app = QtWidgets.QApplication([])
	path = QtWidgets.QFileDialog.getExistingDirectory(caption=caption)
	app.closeAllWindows()
	app.exit()

	return str(path)

def average_same_widths(pixels):
	'''
	Decides which pixels need to be averaged, then average them.
	pixels : dictionary
		Dictionary of format {folder1: device1, folder2: device2...}
	'''
	widths = []

	#dictionary with keys being width and values being arrays of that width
	devicesByWidth = {}

	#container for devices of same width that will be deleted after following for loop
	#this is because these devices will be replaced with the averaged result
	pixelsToDel = [] 

	for pixel in pixels:
		pixelDevice = pixels[pixel]
		currentWidth = pixelDevice.WdL
		if currentWidth not in devicesByWidth: #start new entry if width not yet in dictionary
			devicesByWidth[currentWidth] = [pixelDevice]
		else: 
			devicesByWidth[currentWidth].append(pixelDevice)
			pixelsToDel.append(pixel)
		widths.append(pixelDevice.WdL)

	for pixelToDel in pixelsToDel: #delete duplicates
		del pixels[pixelToDel]
	#produces a list of tuples. tuple index 0 is unique width value, index 1 is occurences of that value
	#this is sorted from most to least occurring
	widthCounts = Counter(widths).most_common()  
	
	while widthCounts[0][1] > 1: #while we still have to deal with averaging more devices
		currentWidth = widthCounts[0][0]
		devices = devicesByWidth[currentWidth]	
		peak_gms = [] #list of gms of devices of same width
		VgVts = [] #list of VgVts of devices of same width
		for device in devices:
			peak_gms.append(device.peak_gm)
			VgVts.append(device.VgVts)

		#get averaged arrays
		gmsMeans = [np.mean([el for el in sublist if el < 0] or 0) for sublist in peak_gms]
		# gmsMeans = np.mean(peak_gms)
		VgVtsMeans = [np.mean([el for el in sublist if el < 0] or 0) for sublist in VgVts]
		widthCounts.pop(0) #remove this width from list since we are done
		
		#replace device with averaged result
		for pixel in pixels:
			if pixels[pixel] == devices[0]:
				pixels[pixel].peak_gm = gmsMeans
				pixels[pixel].VgVts = VgVtsMeans
				break
	return pixels

def average(dimDict={}, path='', thickness=40e-9, plot=True, text_browser=None):
	'''
	averages data in this particular path (for folders 'avg')

	path: str
		string path to folder '.../avg'. Note Windows path are of form r'Path_name'

	thickness : float
		approximate film thickness. Standard polymers (for Raj) are ~40 nm

	plot : bool
		Whether to plot or not. Not plotting is very fast!


	Returns
	-------
	pixels : dict of OECT
		Contains the various OECT class devices

	Id_Vg : pandas dataframe
		Contains the averaged Id vs Vg (drain current vs gate voltage, transfer)

	Id_Vd : pandas dataframe
		Contains the averaged Id vs Vg (drain current vs drain voltages, output)

	'''

	if not path:
		path = file_open(caption='Select avg subfolder')
		print('Loading from', path)

	filelist = os.listdir(path)

	# removes all but the folders in pixel_names
	f = filelist[:]
	for k in filelist:
		try:
			sub_num = int(k)
		except:
			print('Ignoring', k)
			f.remove(k)
	filelist = f[:]
	del f

	paths = [os.path.join(path, name) for name in filelist]

	# removes random files instead of the sub-folders
	for p in paths:
		if not os.path.isdir(p):
			paths.remove(p)

	pixels = {}
	# loads all the folders
	for p, f in zip(paths, filelist):
		dv = loadOECT(dimDict, p, params={'d': thickness}, gm_plot=plot, plot=plot, text_browser=text_browser)
		pixels[f] = dv

	# average Id-Vg
	Id_Vg = []
	first_pxl = pixels[list(pixels.keys())[0]]

	for dv in pixels:

		if not any(Id_Vg):

			Id_Vg = pixels[dv].transfers.values

		else:

			Id_Vg += pixels[dv].transfers.values

	Id_Vg /= len(pixels)
	Id_Vg = pd.DataFrame(data=Id_Vg)

	try:
		Id_Vg = Id_Vg.set_index(first_pxl.transfers.index)
	except:
		Id_Vg = Id_Vg.set_index(pixels[list(pixels.keys())[-1]])

	# find gm of the average
	temp_dv = oect.OECT(dimDict, paths[0], {'d': thickness})
	_gm_fwd, _gm_bwd = temp_dv._calc_gm(Id_Vg)
	Id_Vg['gm_fwd'] = _gm_fwd
	if not _gm_bwd.empty:
		Id_Vg['gm_bwd'] = _gm_bwd

	Id_Vg = Id_Vg.rename(columns={0: 'Id average'})  # fix a naming bug

	if temp_dv.reverse:
		Id_Vg.reverse = True
		Id_Vg.rev_point = temp_dv.rev_point

	# average Id-Vd at max Vd
	Id_Vd = []

	# finds column corresponding to lowest voltage (most doping), but these are strings
	idx = np.argmin(np.array([float(i) for i in first_pxl.outputs.columns]))
	volt = first_pxl.outputs.columns[idx]

	for dv in pixels:

		if not any(Id_Vd):

			Id_Vd = pixels[dv].outputs[volt].values

		else:

			Id_Vd += pixels[dv].outputs[volt].values

	Id_Vd /= len(pixels)
	Id_Vd = pd.DataFrame(data=Id_Vd)

	try:
		Id_Vd = Id_Vd.set_index(pixels[list(pixels.keys())[0]].outputs[volt].index)
	except:
		Id_Vd = Id_Vd.set_index(pixels[list(pixels.keys())[-1]].outputs[volt].index)

	if plot:
		fig = oect_plot.plot_transfer_avg(Id_Vg, temp_dv.WdL)
		fig.savefig(path + r'\transfer_avg.tif', format='tiff')
		fig = oect_plot.plot_output_avg(Id_Vd)
		fig.savefig(path + r'\output_avg.tif', format='tiff')
	return pixels, Id_Vg, Id_Vd, temp_dv.WdL
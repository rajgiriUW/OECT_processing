import numpy as np
import os
import pandas as pd
import streamlit as st
import sys
from matplotlib import pyplot as plt

import oect_processing as oectp
from oect_processing.oect_utils import oect_plot
from oect_processing.oect_utils.oect_load import uC_scale

# sys.path.insert(0, os.path.abspath('..'))
# os.chdir('..')
# os.chdir('..')

TEST_DATA = r'oect_processing/notebooks/test_data_manufactured'

st.set_page_config(page_title='OECT Processing')
st.title('OECT processing')
st.header('Rajiv Giridharagopal, Ph.D.')
st.subheader('University of Washington, rgiri@uw.edu')

with st.expander('Quick Guide'):
    st.write('''
    On the sidebar, input the windows path (copy-paste from Explorer) into "Device Folder". 
    The drop-down menu processes individual folders of data. In the third box, select the pixels you want to use for calculating $\mu_C*$. 
    You need at least 2 pixels. The thickness values are grabbed from .cfg files in each folder. You can override that below.
    Lastly, if you scroll down, you can right-click any graph and save it locally.''')

st.sidebar.header('Load data')


# device = st.sidebar.file_uploader('Select files', accept_multiple_files=True, help='Select all files for a given device')


def file_selector(folder_path=TEST_DATA):
    filenames = []
    for name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, name)):
            filenames.append(name)

    selected_filename = st.sidebar.selectbox('Select a pixel in this folder', filenames)

    st.sidebar.write('Number of devices is `%s`' % len(filenames))

    return os.path.join(folder_path, selected_filename)


def create_pixel_dict(selected_pixels, pixel_paths, thickness):
    pixels = {}
    for s in selected_pixels:
        if thickness:
            dv = oectp.OECT(pixel_paths[s], params={'d': thickness}, options={'Average': True})
        else:
            dv = oectp.OECT(pixel_paths[s], options={'Average': True})
        pixels[s] = dv

    return pixels


device_folder = st.sidebar.text_input('Device folder (only works locally)')
if not device_folder:
    device_folder = TEST_DATA
pixel_folder = file_selector(device_folder)

# Paths to subfolders for each pixel
pixel_paths = {}
paths = [os.path.join(device_folder, name) for name in os.listdir(device_folder)]
folders = []
for p in paths:
    if not os.path.isdir(p):
        paths.remove(p)
    else:
        k = p.split('\\')[-1]
        folders.append(k)
        pixel_paths[k] = p

st.write('Pixel folder is `%s`' % pixel_folder)
st.write('Device folder is `%s`' % device_folder)

st.header('Pixel parameters')
thickness = st.text_input('Thickness (nm), leave blank to use .cfg files')
try:
    thickness = float(thickness) * 1e-9
except:
    thickness = None

selected_pixels = st.sidebar.multiselect('Select pixels for device calculation', folders,
                                         help='Choose which pixels to use in finding $\mu C^*$')

use_spl = st.sidebar.toggle('Use spline values?')

if len(selected_pixels) < 2:
    st.sidebar.markdown('**Must select at least two pixels to generate uC* curve**')

else:
    pixels = create_pixel_dict(selected_pixels, pixel_paths, thickness)
    device = oectp.OECTDevice(pixels=pixels, params={'thickness': thickness, 'L': 10e-6},
                              options={'plot': [False, False], 'verbose': False,
                                       'spline': use_spl})

    # Update sidebar display
    df = pd.DataFrame(index=selected_pixels)
    df['W'] = device.W
    df['gms'] = device.gms
    df['Vt'] = device.Vt
    df['VgVts'] = device.Vg_Vt
    df['Wd/L (nm)'] = np.round(device.WdL * 1e9, 0)
    st.sidebar.write(df)
    st.sidebar.write('Without y-offset fit:')
    st.sidebar.write('$\mu C^*$ = ', '$' + str(np.round(device.uC_0) * 1e-2)[1:-1] + '$', ' $Fcm^{-1}V^{-1}s^{-1}$')
    st.sidebar.write('With y-offset fit:')
    st.sidebar.write('$\mu C^*$ = ', '$' + str(np.round(device.uC[1]) * 1e-2) + '$', ' $Fcm^{-1}V^{-1}s^{-1}$')

dv = oectp.OECT(pixel_folder, params={'d': thickness})
st.write('Width = `%s`' % dv.W)
st.write('Length = `%s`' % dv.L)

dv.calc_gms()
vt_plot, _ = dv.thresh(plot=True)
st.write('Transconductance, $g_m$ (S)')
dv.gms
st.write('Threshold Voltage $V_t$ (V)')
dv.Vts
st.write('Peak $g_m$ (S)')
dv.gm_peaks
st.write('Peak $g_m$ (S), spline')
dv.gm_peaks_spl

st.header('Plots')
use_gm_spl = st.toggle('Use Spline?')
fig = oect_plot.plot_transfers_gm(dv, spline=use_gm_spl)
st.pyplot(fig)

#fig = oect_plot.plot_outputs(dv, sort=True, direction='bwd')
fig = oect_plot.plot_outputs(dv)

st.pyplot(fig)

st.pyplot(vt_plot)

# Run device analysis

_, _, fig = oect_plot.plot_uC(device, savefig=False, average=False)
st.pyplot(fig)

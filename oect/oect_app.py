import streamlit as st
import pandas as pd
import sys, os
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath('..'))
os.chdir('..')
os.chdir('..')

import oect
from oect.oect_utils.oect_load import uC_scale
from oect.oect_utils import oect_plot

st.title('OECT processing')
device_folder = st.sidebar.text_input('Device folder')


def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a pixel in this folder', filenames)

    st.sidebar.write('Number of devices is `%s`' % len(filenames))

    return os.path.join(folder_path, selected_filename)


pixel_folder = file_selector(device_folder)

paths = [os.path.join(device_folder, name) for name in os.listdir(device_folder)]
for p in paths:
    if not os.path.isdir(p):
        paths.remove(p)

st.write('Pixel folder is `%s`' % pixel_folder)
st.write('Device folder is `%s`' % device_folder)

st.header('Pixel parameters')
thickness = st.text_input('Thickness (nm), leave blank to use .cfg files')
try:
    thickness = float(thickness) * 1e-9
except:
    thickness = None

# Run Pixel analysis
if thickness:
    dv = oect.OECT(pixel_folder, params={'d': thickness})
else:
    dv = oect.OECT(pixel_folder)

st.write('Width = `%s`' % dv.W)
st.write('Length = `%s`' % dv.L)

dv.calc_gms()
dv.thresh()
st.write('Transconductance, $g_m$ (S)')
dv.gms
st.write('Threshold Voltage $V_t$ (V)')
dv.Vts
st.write('Peak $g_m$ (S)')
dv.gm_peaks

st.header('Plots')
fig = oect_plot.plot_transfers_gm(dv)
st.pyplot(fig)

fig = oect_plot.plot_outputs(dv, sort=True, direction='bwd')
st.pyplot(fig)

# Run device analysis
device = oect.OECTDevice(device_folder,
                         params={'thickness': thickness},
                         options={'plot': [False, False], 'verbose': False})

_, _, fig = oect_plot.plot_uC(device, savefig=False)
st.pyplot(fig)

# Update sidebar display
df = pd.DataFrame(index= device.W)
df['gms'] = device.gms
df['Vt'] = device.Vt
df['VgVts'] = device.Vg_Vt
df['Wd/L (nm)'] = np.round(device.WdL * 1e9, 0)
st.sidebar.write(df)
st.sidebar.write('$\mu C^*$ = ', '$'+str(np.round(device.uC_0)*1e-2)[1:-1]+'$', ' $Fcm^{-1}V^{-1}s^{-1}$')

#st.multiselect('Multiselect', [1,2,3])

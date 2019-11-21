# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:17:42 2019

@author: Raj
"""

import uvvis
from matplotlib import pyplot as plt
import numpy as np

df01 = r'C:/Users/Raj/SkyDrive/UW Work/Data/Polymer OECT work/DPP3TEG-1/uvvis/20191119/01 - CF 1-0 4 mgmL KPF6/outputs/dopingdata.pkl'
df02 = r'C:/Users/Raj/SkyDrive/UW Work/Data/Polymer OECT work/DPP3TEG-1/uvvis/20191119/02 - CB 2-8 20 mgmL KPF6/outputs/dopingdata.pkl'
df03 = r'C:\Users\Raj\SkyDrive\UW Work\Data\Polymer OECT work\DPP3TEG-1\uvvis\20191119\03 - CB 1-0 10 mgmL KPF6\outputs\dopingdata.pkl'
df04 = r'C:\Users\Raj\SkyDrive\UW Work\Data\Polymer OECT work\DPP3TEG-1\uvvis\20191119\04 - CB 2-8 20 mgmL KCl (sample 01)\outputs\dopingdata.pkl'
df05 = r'C:\Users\Raj\SkyDrive\UW Work\Data\Polymer OECT work\DPP3TEG-1\uvvis\20191119\05 - CF 1-0 4 mgmL KCl\outputs\dopingdata.pkl'
df06 = r'C:\Users\Raj\SkyDrive\UW Work\Data\Polymer OECT work\DPP3TEG-1\uvvis\20191119\06 - CB 2-8 20 mgmL KCl (sample 02)\outputs\dopingdata.pkl'

with open(df01, 'rb') as input:
    df_01 = pickle.load(input)

with open(df02, 'rb') as input:
    df_02 = pickle.load(input)
    
with open(df03, 'rb') as input:
    df_03 = pickle.load(input)
    
with open(df04, 'rb') as input:
    df_04 = pickle.load(input)
    
with open(df05, 'rb') as input:
    df_05 = pickle.load(input)
    
with open(df06, 'rb') as input:
    df_06 = pickle.load(input)

data = [df_01, df_02, df_03, df_04, df_05, df_06]
data_kpf6 = [df_01, df_02, df_03]
data_kcl = [df_04, df_05, df_06]
potential = 0.8
wl_start = 700
wl_stop = 850
tx = df_02.spectra_vs_time[potential][wl_start:wl_stop].index.values

# Banded plot
for d in data:
    d.banded_plot(voltage=potential, wl_start=wl_start, wl_stop=wl_stop)


thicknesses = [110e-9, 40e-9, 80e-9, 40e-9, 110e-9, 40e-9]
labels = ['CF 1:0 4 mg/mL', 'CB 1:4 20 mg/mL', 'CB 1:0 10 mg/mL', 'CB 1:4 20 mg/mL', 'CB 1:0 10 mg/mL', 'CB 1:4 20 mg/mL']
markers = ['o', '^', 'd', 's', 'v', 'X']
colors = ['r', 'k', 'b', 'g', 'xkcd:olive green', 'xkcd:maroon']

fig, ax = plt.subplots(nrows=2, facecolor='white', figsize=(6,8))
ax[1].set_xlabel ('Thickness (nm)')
ax[0].set_ylabel ('Time Constant (s), KPF6')
ax[1].set_ylabel ('Time Constant (s), KCl')
for d, t, l, m, c in zip(data_kpf6, thicknesses, labels, markers, colors):
    ax[0].plot(np.ones(len(d.fits))*t*1e9, d.fits, **{'marker': m, 'color': c, 'linewidth':0})
    #ax[1].plot(l, d.fits, **{'marker': m, 'color': c})
for d, t, l, m, c in zip(data_kcl, thicknesses[3:], labels[3:], markers[3:], colors[3:]):
    ax[1].plot(np.ones(len(d.fits))*t*1e9, d.fits, **{'marker': m, 'color': c, 'linewidth':0})
ax[0].legend(labels=labels[:3])
ax[1].legend(labels=labels[3:])

fig, ax = plt.subplots(nrows=2, facecolor='white', figsize=(6,8))
ax[1].set_xlabel ('Thickness (nm)')
ax[0].set_ylabel ('Rate Constant (Hz), KPF6')
ax[1].set_ylabel ('Rate Constant (Hz), KCl')
for d, t, l, m, c in zip(data_kpf6, thicknesses, labels, markers, colors):
    ax[0].plot(np.ones(len(d.fits))*t*1e9, np.array(d.fits)**-1, **{'marker': m, 'color': c, 'linewidth':0})
    #ax[1].plot(l, d.fits, **{'marker': m, 'color': c})
for d, t, l, m, c in zip(data_kcl, thicknesses[3:], labels[3:], markers[3:], colors[3:]):
    ax[1].plot(np.ones(len(d.fits))*t*1e9, np.array(d.fits)**-1, **{'marker': m, 'color': c, 'linewidth':0})
ax[0].legend(labels=labels[:3])
ax[1].legend(labels=labels[3:])

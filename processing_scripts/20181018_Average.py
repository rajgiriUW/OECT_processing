# -*- coding: utf-8 -*-
"""
@author: Raj
"""

import numpy as np
import OECT, OECT_loading, OECT_plotting
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as cf
import pandas as pd


# fit functions
def line_f(x, a, b):
    return a + b * x


def line_0(x, b):
    'no y-offset --> better log-log fits'
    return b * x


# %%
# avg paths to average together

# No Washing - 10 mg/mL - 4:6 wires
paths_46_nowash = [r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 01\avg',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 02\avg',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180918 - dppdtt 46nowash_4\avg',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180918 - dppdtt 46nowash_3\avg',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181010 - dppdtt 46 nowash 03 (new geom)\avg',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180911 - dppdtt 4-6_nowash_01 devices\avg']

gm_peaks = np.array([])
gm_average = {}
WdLs = np.array([])
VgVts = np.array([])

columns = [str(p) for p in np.arange(len(paths_46_nowash))]

for p, c in zip(paths_46_nowash, columns):
    print(p)
    pixels, Id_Vg, Id_Vd, Wd_L = OECT_loading.average(p, plot=False)

    for x in pixels:
        VgVts = np.append(VgVts, pixels[x].Vts)
        gm_peaks = np.append(gm_peaks, pixels[x].gm_peaks)
        WdLs = np.append(WdLs, Wd_L)

        if len(pixels[x].gm_peaks.values) > 1:
            WdLs = np.append(WdLs, Wd_L)

    gm_average[c] = Id_Vg

fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))
plt.rc('axes', linewidth=4)
plt.rcParams.update({'font.size': 18, 'font.weight': 'bold',
                     'font.sans-serif': 'Arial'})

ax.tick_params(axis='both', length=10, width=3, which='major', top='on')
ax.tick_params(axis='both', length=6, width=3, which='minor', top='on')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax2.tick_params(axis='both', length=10, width=3, which='major')
ax2.tick_params(axis='both', length=6, width=3, which='minor')
ax2.tick_params(axis='y', labelsize=18)

ax2 = ax.twinx()
ax.set_xlabel('Vg (V)', fontweight='bold', fontsize=18, fontname='Arial')
ax.set_ylabel('Id (mA)', fontweight='bold', fontsize=18, fontname='Arial')
ax2.set_ylabel('gm (mS)', rotation=-90, fontweight='bold', fontsize=18, fontname='Arial', labelpad=20)

for k in gm_average:
    p = ax.plot(gm_average[k]['Id average'] * 1000, linestyle='-', linewidth=3)
    ax2.plot(gm_average[k]['gm_fwd'] * 1e3, linestyle='--', linewidth=2, color=p[0]._color)
    ax2.plot(gm_average[k]['gm_bwd'] * 1e3, linestyle='--', linewidth=2, color=p[0]._color)

path = r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\2018_1018,0918 uC aggregate'
fig.savefig(path + r'\aggregate_Iv_averages.tif', format='tiff')

fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))
ax.set_xlabel('Wd/L (nm)', fontweight='bold', fontsize=18, fontname='Arial')
ax.set_ylabel('gm (mS)', fontweight='bold', fontsize=18, fontname='Arial')
ax.plot(WdLs * 1e9, gm_peaks * 1e3, 's', markersize=12)
fig.savefig(path + r'\gm_scatter.tif', format='tiff')
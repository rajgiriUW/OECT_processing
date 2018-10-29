# -*- coding: utf-8 -*-
"""
@author: Raj
"""

import numpy as np
import OECT_loading
from matplotlib import pyplot as plt
from scipy import signal as sps
import pandas as pd


# fit functions
def line_f(x, a, b):
    return a + b * x


def line_0(x, b):
    'no y-offset --> better log-log fits'
    return b * x


def gm_deriv(v, i, method, fit_params={'window': 11, 'polyorder': 2, 'deg': 8}):
    if method is 'sg':
        # Savitsky-Golay method
        if not fit_params['window'] & 1:  # is odd
            fit_params['window'] += 1
        gml = sps.savgol_filter(i.T, window_length=fit_params['window'],
                                polyorder=fit_params['polyorder'], deriv=1,
                                delta=v[2] - v[1])
    elif method is 'raw':
        # raw derivative
        gml = np.gradient(i.flatten(), v[2] - v[1])

    elif method is 'poly':
        # polynomial fit
        funclo = np.polyfit(v, i, fit_params['deg'])
        gml = np.gradient(np.polyval(funclo, v), (v[2] - v[1]))

    else:
        return

    return gml


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
        gm_peaks = np.append(gm_peaks, pixels[x].gm_peaks / pixels[x].d)
        WdLs = np.append(WdLs, Wd_L / pixels[x].d)

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

ax2 = ax.twinx()
ax.set_xlabel('Vg (V)', fontweight='bold', fontsize=18, fontname='Arial')
ax.set_ylabel('Id (mA)', fontweight='bold', fontsize=18, fontname='Arial')
ax2.set_ylabel('gm (mS)', rotation=-90, fontweight='bold', fontsize=18, fontname='Arial', labelpad=20)
ax2.tick_params(axis='both', length=10, width=3, which='major')
ax2.tick_params(axis='both', length=6, width=3, which='minor')
ax2.tick_params(axis='y', labelsize=18)

for k in gm_average:
    p = ax.plot(gm_average[k]['Id average'] * 1000, linestyle='-', linewidth=3)
    ax2.plot(gm_average[k]['gm_fwd'] * 1e3, linestyle='--', linewidth=2, color=p[0]._color)
    ax2.plot(gm_average[k]['gm_bwd'] * 1e3, linestyle='--', linewidth=2, color=p[0]._color)

path = r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\2018_1018,0918 uC aggregate'
fig.savefig(path + r'\aggregate_no_wash_Iv_averages.tif', format='tiff')

# %%
# Plotting only the average Id-Vg for 0.01 V step data
paths_46_nowash = [r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 01\avg',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 02\avg',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181010 - dppdtt 46 nowash 03 (new geom)\avg']

gm_peaks = np.array([])
IdVg_average = pd.DataFrame()
gm_average = np.array([])
WdLs = np.array([])
VgVts = np.array([])

columns = [str(p) for p in np.arange(len(paths_46_nowash))]

for p, c in zip(paths_46_nowash, columns):
    print(p)
    pixels, Id_Vg, Id_Vd, Wd_L = OECT_loading.average(p, plot=False)

    for x in pixels:
        VgVts = np.append(VgVts, pixels[x].Vts)
        gm_peaks = np.append(gm_peaks, pixels[x].gm_peaks / pixels[x].d)
        WdLs = np.append(WdLs, Wd_L / pixels[x].d)

        if len(pixels[x].gm_peaks.values) > 1:
            WdLs = np.append(WdLs, Wd_L)
    if IdVg_average.empty:

        IdVg_average = pd.DataFrame(data=Id_Vg['Id average'].values, index=Id_Vg.index, columns=[c])

    else:

        s = pd.DataFrame(data=Id_Vg['Id average'].values, index=Id_Vg.index, columns=[c])
        IdVg_average = pd.concat([IdVg_average, s], axis=1)

IdVg_average['avg'] = IdVg_average.mean(axis=1)
window = 11
polyorder = 2
deg = 8

# Get gm
v = IdVg_average.index.values
i = IdVg_average['avg'].values
gml = gm_deriv(v, i, 'sg', {'window': window, 'polyorder': polyorder, 'deg': deg})
gml = np.abs(gml)

fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))
plt.rc('axes', linewidth=4)
plt.rcParams.update({'font.size': 18, 'font.weight': 'bold',
                     'font.sans-serif': 'Arial'})

ax.tick_params(axis='both', length=10, width=3, which='major', top='on')
ax.tick_params(axis='both', length=6, width=3, which='minor', top='on')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

ax2 = ax.twinx()
ax.set_xlabel('Vg (V)', fontweight='bold', fontsize=18, fontname='Arial')
ax.set_ylabel('Id (mA)', fontweight='bold', fontsize=18, fontname='Arial')
ax2.set_ylabel('gm (mS)', rotation=-90, fontweight='bold', fontsize=18, fontname='Arial', labelpad=20)
ax2.tick_params(axis='both', length=10, width=3, which='major')
ax2.tick_params(axis='both', length=6, width=3, which='minor')
ax2.tick_params(axis='y', labelsize=18)
ax.set_title('Id Average and gm average')

p = ax.plot(IdVg_average['avg'] * 1000, linestyle='-', linewidth=3)
ax2.plot(IdVg_average.index.values, gml * 1e3, linestyle='--', linewidth=2, color=p[0]._color)

path = r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\2018_1018,0918 uC aggregate'
fig.savefig(path + r'\aggregate_nowash_Iv_average_total.tif', format='tiff')

# %%
# Scatter plot
paths_46_nowash_uC = [r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 01\uC',
                      r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 02\uC',
                      r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181010 - dppdtt 46 nowash 03 (new geom)\uC',
                      r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180918 - dppdtt 46nowash_4\uC',
                      r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180918 - dppdtt 46nowash_3\uC']

columns = [str(p) for p in np.arange(len(paths_46_nowash_uC))]

gm_peaks = np.array([])
gm_average = {}
WdLs = np.array([])
VgVts = np.array([])

for p, c in zip(paths_46_nowash_uC, columns):
    print(p)
    pixels, uC = OECT_loading.uC_scale(p, plot=False)

    for x in pixels:
        gm_peaks = np.append(gm_peaks, pixels[x].gm_peaks / (pixels[x].d * pixels[x].W * pixels[x].L))
        WdLs = np.append(WdLs, pixels[x].WdL)

        if len(pixels[x].gm_peaks.values) > 1:
            WdLs = np.append(WdLs, pixels[x].WdL)

fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))
ax.set_xlabel('Wd/L (nm)', fontweight='bold', fontsize=18, fontname='Arial')
ax.set_ylabel('gm / V (mS / um^3)', fontweight='bold', fontsize=18, fontname='Arial')
ax.plot(WdLs * 1e9, gm_peaks * 1e3 / (1e6) ** 3, 's', markersize=12)
fig.savefig(path + r'\gm_nowash_scatter.tif', format='tiff')

# %%
# Washed wire sample
# "good" data
paths_46_wash = [r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 wash 03\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 wash 04\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181005 - dppdtt 46 wash 02 (new geom)\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181005 - dppdtt 46 wash 01 (orig geom)\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180720 - DPP DTT PS devices\02_wash\uC']

gm_peaks = np.array([])
gm_average = {}
WdLs = np.array([])
VgVts = np.array([])

columns = [str(p) for p in np.arange(len(paths_46_wash))]

for p, c in zip(paths_46_wash, columns):
    print(p)
    pixels, Id_Vg, Id_Vd, Wd_L = OECT_loading.average(p, plot=False)

    for x in pixels:
        VgVts = np.append(VgVts, pixels[x].Vts)
        gm_peaks = np.append(gm_peaks, pixels[x].gm_peaks / pixels[x].d)
        WdLs = np.append(WdLs, Wd_L / pixels[x].d)

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

ax2 = ax.twinx()
ax.set_xlabel('Vg (V)', fontweight='bold', fontsize=18, fontname='Arial')
ax.set_ylabel('Id (mA)', fontweight='bold', fontsize=18, fontname='Arial')
ax2.set_ylabel('gm (mS)', rotation=-90, fontweight='bold', fontsize=18, fontname='Arial', labelpad=20)
ax2.tick_params(axis='both', length=10, width=3, which='major')
ax2.tick_params(axis='both', length=6, width=3, which='minor')
ax2.tick_params(axis='y', labelsize=18)

for k in gm_average:
    p = ax.plot(gm_average[k]['Id average'] * 1000, linestyle='-', linewidth=3)
    ax2.plot(gm_average[k]['gm_fwd'] * 1e3, linestyle='--', linewidth=2, color=p[0]._color)
    try:
        ax2.plot(gm_average[k]['gm_bwd'] * 1e3, linestyle='--', linewidth=2, color=p[0]._color)
    except:
        print('No backward trace!')

path = r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\2018_1018,0918 uC aggregate'
fig.savefig(path + r'\aggregate_wash_Iv_averages.tif', format='tiff')

# %%
# "Best" washed data
# IdVg averages
paths_46_wash = [r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 wash 03\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 wash 04\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181005 - dppdtt 46 wash 02 (new geom)\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181005 - dppdtt 46 wash 01 (orig geom)\uC']

gm_peaks = np.array([])
IdVg_average = pd.DataFrame()
gm_average = np.array([])
WdLs = np.array([])
VgVts = np.array([])

columns = [str(p) for p in np.arange(len(paths_46_wash))]

for p, c in zip(paths_46_wash, columns):
    print(p)
    pixels, Id_Vg, Id_Vd, Wd_L = OECT_loading.average(p, plot=False)

    for x in pixels:
        VgVts = np.append(VgVts, pixels[x].Vts)
        gm_peaks = np.append(gm_peaks, pixels[x].gm_peaks / pixels[x].d)
        WdLs = np.append(WdLs, Wd_L / pixels[x].d)

        if len(pixels[x].gm_peaks.values) > 1:
            WdLs = np.append(WdLs, Wd_L)
    if IdVg_average.empty:

        IdVg_average = pd.DataFrame(data=Id_Vg['Id average'].values, index=Id_Vg.index, columns=[c])

    else:

        s = pd.DataFrame(data=Id_Vg['Id average'].values, index=Id_Vg.index, columns=[c])
        IdVg_average = pd.concat([IdVg_average, s], axis=1)

IdVg_average['avg'] = IdVg_average.mean(axis=1)
window = 11
polyorder = 2
deg = 8

# Get gm
v = IdVg_average.index.values
i = IdVg_average['avg'].values
gml = gm_deriv(v, i, 'sg', {'window': window, 'polyorder': polyorder, 'deg': deg})
gml = np.abs(gml)

fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))
plt.rc('axes', linewidth=4)
plt.rcParams.update({'font.size': 18, 'font.weight': 'bold',
                     'font.sans-serif': 'Arial'})

ax.tick_params(axis='both', length=10, width=3, which='major', top='on')
ax.tick_params(axis='both', length=6, width=3, which='minor', top='on')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

ax2 = ax.twinx()
ax.set_xlabel('Vg (V)', fontweight='bold', fontsize=18, fontname='Arial')
ax.set_ylabel('Id (mA)', fontweight='bold', fontsize=18, fontname='Arial')
ax2.set_ylabel('gm (mS)', rotation=-90, fontweight='bold', fontsize=18, fontname='Arial', labelpad=20)
ax2.tick_params(axis='both', length=10, width=3, which='major')
ax2.tick_params(axis='both', length=6, width=3, which='minor')
ax2.tick_params(axis='y', labelsize=18)
ax.set_title('Id Average and gm average')

p = ax.plot(IdVg_average['avg'] * 1000, linestyle='-', linewidth=3)
ax2.plot(IdVg_average.index.values, gml * 1e3, linestyle='--', linewidth=2, color=p[0]._color)

path = r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\2018_1018,0918 uC aggregate'
fig.savefig(path + r'\aggregate_wash_Iv_average_total.tif', format='tiff')

# %%
# Washed data
# gm scatter

paths_46_wash_uC = [r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 wash 03\uC',
                    r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 wash 04\uC',
                    r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181005 - dppdtt 46 wash 02 (new geom)\uC',
                    r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181005 - dppdtt 46 wash 01 (orig geom)\uC']

columns = [str(p) for p in np.arange(len(paths_46_wash_uC))]

gm_peaks = np.array([])
gm_average = {}
WdLs = np.array([])
VgVts = np.array([])

for p, c in zip(paths_46_wash_uC, columns):
    print(p)
    pixels, uC = OECT_loading.uC_scale(p, plot=False)

    for x in pixels:
        gm_peaks = np.append(gm_peaks, pixels[x].gm_peaks / (pixels[x].d * pixels[x].W * pixels[x].L))
        WdLs = np.append(WdLs, pixels[x].WdL)

        if len(pixels[x].gm_peaks.values) > 1:
            WdLs = np.append(WdLs, pixels[x].WdL)

fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))
ax.set_xlabel('Wd/L (nm)', fontweight='bold', fontsize=18, fontname='Arial')
ax.set_ylabel('gm / V (mS / um^3)', fontweight='bold', fontsize=18, fontname='Arial')
ax.plot(WdLs * 1e9, gm_peaks * 1e3 / (1e6) ** 3, 's', markersize=12)
fig.savefig(path + r'\gm_wash_scatter.tif', format='tiff')

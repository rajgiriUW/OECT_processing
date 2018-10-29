# -*- coding: utf-8 -*-
"""
@author: Raj
"""

import numpy as np
import OECT, OECT_loading, OECT_plotting
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as cf


# fit functions
def line_f(x, a, b):
    return a + b * x


def line_0(x, b):
    'no y-offset --> better log-log fits'
    return b * x


# %%
# uC paths to average together

# No Washing - 10 mg/mL - 4:6 wires
paths_46_nowash = [r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 01\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 02\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180918 - dppdtt 46nowash_4\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180918 - dppdtt 46nowash_3\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181010 - dppdtt 46 nowash 03 (new geom)\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180911 - dppdtt 4-6_nowash_01 devices\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180824 - dppdtt devices\wire_nowash\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180720 - DPP DTT PS devices\01_noWash\uC']

Wd_Ls = np.array([])
gm_peaks = np.array([])
VgVts = np.array([])

fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))

for p in paths_46_nowash:
    print(p)
    _, uC = OECT_loading.uC_scale(p, plot=False, add_avg_pixels=True)
    Wd_Ls = np.append(Wd_Ls, uC.Wd_L)
    VgVts = np.append(VgVts, uC.Vg_Vt)
    gm_peaks = np.append(gm_peaks, uC.gms)
    ax.plot(uC.Wd_L * uC.Vg_Vt * 1e2, uC.gms * 1e3, 's', markersize=8)

uC_0, _ = cf(line_0, Wd_Ls * VgVts, gm_peaks)
uC, _ = cf(line_f, Wd_Ls * VgVts, gm_peaks)
ax.set_title('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
ax.set_xlabel('Wd/L * (Vg-Vt) (cm*V)')
ax.set_ylabel('gm (mS)')

dv = {'Wd_L': Wd_Ls, 'Vg_Vt': VgVts, 'gms': gm_peaks, 'uC': uC, 'uC_0': uC_0,
      'folder': r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\2018_1018,0918 uC aggregate'}

OECT_plotting.plot_uC(dv, label='wire_nowash_all')

# %%
# "Good" subset
paths_46_nowash = [r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 01\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 02\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181010 - dppdtt 46 nowash 03 (new geom)\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180918 - dppdtt 46nowash_4\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180918 - dppdtt 46nowash_3\uC']

Wd_Ls = np.array([])
gm_peaks = np.array([])
VgVts = np.array([])

fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))

for p in paths_46_nowash:
    print(p)
    _, uC = OECT_loading.uC_scale(p, plot=False, add_avg_pixels=True)
    Wd_Ls = np.append(Wd_Ls, uC.Wd_L)
    VgVts = np.append(VgVts, uC.Vg_Vt)
    gm_peaks = np.append(gm_peaks, uC.gms)
    ax.plot(uC.Wd_L * uC.Vg_Vt * 1e2, uC.gms * 1e3, 's', markersize=8)

uC_0, _ = cf(line_0, Wd_Ls * VgVts, gm_peaks)
uC, _ = cf(line_f, Wd_Ls * VgVts, gm_peaks)
ax.set_title('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
ax.set_xlabel('Wd/L * (Vg-Vt) (cm*V)')
ax.set_ylabel('gm (mS)')

dv = {'Wd_L': Wd_Ls, 'Vg_Vt': VgVts, 'gms': gm_peaks, 'uC': uC, 'uC_0': uC_0,
      'folder': r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\2018_1018,0918 uC aggregate'}

OECT_plotting.plot_uC(dv, label='wire_nowash_good')

# %%
# "Best" subset
paths_46_nowash = [r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 01\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 nowash 02\uC',
                   r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181010 - dppdtt 46 nowash 03 (new geom)\uC']

Wd_Ls = np.array([])
gm_peaks = np.array([])
VgVts = np.array([])

fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))

for p in paths_46_nowash:
    print(p)
    _, uC = OECT_loading.uC_scale(p, plot=False, add_avg_pixels=True)
    Wd_Ls = np.append(Wd_Ls, uC.Wd_L)
    VgVts = np.append(VgVts, uC.Vg_Vt)
    gm_peaks = np.append(gm_peaks, uC.gms)
    ax.plot(uC.Wd_L * uC.Vg_Vt * 1e2, uC.gms * 1e3, 's', markersize=8)

uC_0, _ = cf(line_0, Wd_Ls * VgVts, gm_peaks)
uC, _ = cf(line_f, Wd_Ls * VgVts, gm_peaks)
ax.set_title('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
ax.set_xlabel('Wd/L * (Vg-Vt) (cm*V)')
ax.set_ylabel('gm (mS)')

dv = {'Wd_L': Wd_Ls, 'Vg_Vt': VgVts, 'gms': gm_peaks, 'uC': uC, 'uC_0': uC_0,
      'folder': r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\2018_1018,0918 uC aggregate'}

OECT_plotting.plot_uC(dv, label='wire_nowash_best')

# %%
# Washed Wire data
# "Best data"
paths_46_wash = [r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 wash 03\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 wash 04\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181005 - dppdtt 46 wash 02 (new geom)\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181005 - dppdtt 46 wash 01 (orig geom)\uC']

Wd_Ls = np.array([])
gm_peaks = np.array([])
VgVts = np.array([])

fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))

for p in paths_46_wash:
    print(p)
    _, uC = OECT_loading.uC_scale(p, plot=False, add_avg_pixels=True)
    Wd_Ls = np.append(Wd_Ls, uC.Wd_L)
    VgVts = np.append(VgVts, uC.Vg_Vt)
    gm_peaks = np.append(gm_peaks, uC.gms)
    ax.plot(uC.Wd_L * uC.Vg_Vt * 1e2, uC.gms * 1e3, 's', markersize=8)

uC_0, _ = cf(line_0, Wd_Ls * VgVts, gm_peaks)
uC, _ = cf(line_f, Wd_Ls * VgVts, gm_peaks)
ax.set_title('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
ax.set_xlabel('Wd/L * (Vg-Vt) (cm*V)')
ax.set_ylabel('gm (mS)')

dv = {'Wd_L': Wd_Ls, 'Vg_Vt': VgVts, 'gms': gm_peaks, 'uC': uC, 'uC_0': uC_0,
      'folder': r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\2018_1018,0918 uC aggregate'}

OECT_plotting.plot_uC(dv, label='wire_wash_best')

# %%
# "Good" data

paths_46_wash = [r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 wash 03\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181018 - DPPDTT 46 wash 04\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181005 - dppdtt 46 wash 02 (new geom)\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20181005 - dppdtt 46 wash 01 (orig geom)\uC',
                 r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\20180720 - DPP DTT PS devices\02_wash\uC']

Wd_Ls = np.array([])
gm_peaks = np.array([])
VgVts = np.array([])

fig, ax = plt.subplots(facecolor='white', figsize=(10, 8))

for p in paths_46_wash:
    print(p)
    _, uC = OECT_loading.uC_scale(p, plot=False, add_avg_pixels=True)
    Wd_Ls = np.append(Wd_Ls, uC.Wd_L)
    VgVts = np.append(VgVts, uC.Vg_Vt)
    gm_peaks = np.append(gm_peaks, uC.gms)
    ax.plot(uC.Wd_L * uC.Vg_Vt * 1e2, uC.gms * 1e3, 's', markersize=8)

uC_0, _ = cf(line_0, Wd_Ls * VgVts, gm_peaks)
uC, _ = cf(line_f, Wd_Ls * VgVts, gm_peaks)
ax.set_title('uC* = ' + str(uC_0 * 1e-2) + ' F/cm*V*s')
ax.set_xlabel('Wd/L * (Vg-Vt) (cm*V)')
ax.set_ylabel('gm (mS)')

dv = {'Wd_L': Wd_Ls, 'Vg_Vt': VgVts, 'gms': gm_peaks, 'uC': uC, 'uC_0': uC_0,
      'folder': r'C:\Users\Raj\OneDrive\UW Work\Data\DPP-DTT\_devices\2018_1018,0918 uC aggregate'}

OECT_plotting.plot_uC(dv, label='wire_wash_good')

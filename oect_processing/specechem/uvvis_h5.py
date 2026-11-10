from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import integrate as spint
from .uvvis import UVVis


def save_h5(data, filename):
    '''
    Saves UVVis data to an HDF5 file (.h5).

    Parameters
    ----------
    data : UVVis
        UVVis object containing spectra_vs_time, current, charge, and potentials.
    filename : str or Path
        File path to save to.
    '''

    if isinstance(filename, str):
        filename = Path(filename)

    with h5py.File(filename, 'w') as f:
        f.create_dataset('potentials', data=data.potentials)

        try:
            f.create_dataset('charge', data=data.charge.values[0])
        except:
            pass

        for p in data.spectra_vs_time:
            df = data.spectra_vs_time[p]
            grp = f.create_group(str(p))
            grp.create_dataset('data', data=df.values)
            grp.create_dataset('index', data=df.index.values)
            grp.create_dataset('columns', data=df.columns.values.astype(float))

        try:
            grp = f.create_group('current')
            grp.create_dataset('data', data=data.current.values)
            grp.create_dataset('index', data=data.current.index.values)
            grp.create_dataset('columns', data=data.current.columns.values.astype(float))
        except:
            pass

    return


def convert_h5(h5file):
    '''
    Loads a saved HDF5 file and returns a UVVis object.

    Parameters
    ----------
    h5file : str or Path
        Path to the HDF5 file to load.

    Returns
    -------
    UVVis
        UVVis object populated with spectra_vs_time, current, charge, and potentials.
    '''
    data = UVVis(None, None, None)

    with h5py.File(h5file, 'r') as file:
        data.potentials = file['potentials'][()]

        non_data = {'current', 'charge', 'potentials'}
        folders = [k for k in file.keys() if k not in non_data]

        try:
            folders_num = [float(p) for p in folders]
        except:  # for old 'x-1.0V' style, crops 'x'
            folders_num = [float(p[1:]) for p in folders]

        df_dict = {}
        for v, n in zip(folders, folders_num):
            grp = file[v]
            df = pd.DataFrame(data=grp['data'][()],
                               index=grp['index'][()],
                               columns=grp['columns'][()])
            df.index.name = 'Wavelength (nm)'
            df.columns.name = 'Time (s)'
            df_dict[n] = df
            data.tx = np.round(grp['columns'][()], 2)

        data.spectra_vs_time = df_dict

        grp = file['current']
        data.current = pd.DataFrame(data=grp['data'][()],
                                    index=grp['index'][()],
                                    columns=grp['columns'][()])
        data.current.index.name = 'Time (s)'
        data.current.columns.name = 'Potential (V)'

        data.charge = pd.DataFrame(data=file['charge'][()],
                                   index=data.potentials.T)

    return data

'''

data = uvvis.UVVis(steps, specs, volts)

data.time_dep_spectra(specfiles=specs, droptimes=droptimes) 
data.single_wl_time(wavelength=wavelength, potential=potential)
data.current_vs_time(stepfiles=steps)
data.spec_echem_voltage(time=0)

'''


def convert_h5_python(h5_path):
    '''
    Converts h5 from the format saved in the Python version of spectroelectrochemistry 
    
    The structure of that file format is like this:
    /
    ├ 0
      -
      ├ absorbance_vs_time
      ├ counts_vs_time
      ├ dark
      ├ echem
        -----
        ├ current
        ├ potential
        ├ time
      ├ reference
      ├ time
      ├ time_spectrometer

    Parameters
    ----------
    h5file : str
        Path to the .h5 file on disk.

    Returns
    -------
    Fake object called SEC class or a UVVis object if you have the package installed

    '''
    
    class SEC:
        pass
    try:
        data = UVVis(None, None, None)
    except:
        print('oect_processing package not installed, using fake Class object')
        data = SEC
    current = {}
    with h5py.File(h5_path, 'r') as h5_file:
        
        # Sorts 0, 1, 10 otherwise 1, 10, 11 occurs due to ASCII
        top_groups = sorted(h5_file.keys())
        keys = sorted(top_groups[:-1], key=int)
        data.potentials = []
        wavelength = h5_file['wavelength'][()]
        data.spectra_vs_time = {}
        
        for k in keys:
            # Spectra vs time dictionary

            potential = h5_file[k]['echem']['potential'][0]
            potential = np.round(potential, 2) 
            times = h5_file[k]['echem']['time'][()]
            data.spectra_vs_time[potential] = pd.DataFrame(data = h5_file[k]['absorbance_vs_time'],
                                                           index = wavelength,
                                                           columns = times)
            current[potential] = h5_file[k]['echem']['current'][()]
            data.potentials.append(potential)
            
        data.current = pd.DataFrame(data=current, index=times)
        data.charge = pd.DataFrame(columns=data.current.columns, index=[0])
        data.charge.columns.name = 'Potential (V)'
        for p in data.current:
            data.charge[p] = spint.trapezoid(data.current[p].values, x=data.current.index.values)
    
    return data
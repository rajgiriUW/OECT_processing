from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .uvvis import UVVis


def save_h5(data, filename):
    '''
    Saves the to two HDF5 files (.h5)
    
    :param data: data to save to h5
    :type data: dataframe
    
    :param filename: file path to save to
    :type filename: str
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
    Converts a saved hdf5 to uvvis Class format
    
    axis0 = time
    axis1 = wavelength
    block0_items
    
    :param h5file:
    :type h5file: str
    
    :returns: h5file in uvvis class format
    :rtype: uvvis class object
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

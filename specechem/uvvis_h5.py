def save_h5(data, filename):
    '''
    Saves the to two HDF5 files (.h5)
    '''

    if isinstance(filename, str):
        filename = Path(filename)

    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('potentials', (len(data.potentials),))
        dset[:] = data.potentials[:]
        try:
            dset = f.create_dataset('charge', (len(data.charge.values[0]),))
            dset[:] = data.charge.values[0][:]
        except:
            pass
    f.close()

    for p in data.spectra_vs_time:
        data.spectra_vs_time[p].to_hdf(filename, key=str(p), mode='a')

    try:
        data.current.to_hdf(filename, key='current', mode='a')
    except:
        pass

    return


def convert_h5(h5file):
    '''
    Converts a saved hdf5 to uvvis Class format
    
    axis0 = time
    axis1 = wavelength
    block0_items
    '''
    data = uv_vis(None, None, None)
    file = h5py.File(h5file, 'r')
    data.potentials = file['potentials'][()]

    folders = []
    for f in file:
        folders.append(f)
    non_potentials = ['current', 'charge', 'potentials']
    _fold_temp = [c if c not in non_potentials else None for c in folders]
    folders = [c for c in _fold_temp if (c)]

    try:
        folders_num = [float(p) for p in folders[:]]
    except:  # for old 'x-1.0V' style, crops 'x'
        folders_num = [float(p[1:]) for p in folders[:]]

    # The spectra_vs_time data
    df_dict = {}
    for v, n in zip(folders, folders_num):
        try:
            spec_file = file[v]
        except:
            p = 'x' + v
            spec_file = file[p]

        df = pd.DataFrame(data=spec_file['block0_values'][()],
                          index=spec_file['axis1'][()],
                          columns=spec_file['axis0'])
        df.index.name = 'Wavelength (nm)'
        df.columns.name = 'Time (s)'
        df_dict[n] = df
        data.tx = np.round(spec_file['axis0'], 2)

    data.spectra_vs_time = df_dict

    # Now get the current data
    current = pd.DataFrame(data=file['current']['block0_values'][()],
                           index=file['current']['axis1'][()],
                           columns=file['current']['axis0'])
    current.index.name = 'Time (s)'
    current.columns.name = 'Potential (V)'

    data.current = current

    data.charge = pd.DataFrame(data=file['charge'][()],
                               index=data.potentials.T)

    file.close()

    return data

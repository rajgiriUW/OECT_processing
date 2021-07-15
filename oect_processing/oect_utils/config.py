import configparser
import pathlib

def make_config(path):
    '''
    If a config file does not exist, this will generate one automatically.
    
    '''
    config = configparser.ConfigParser()
    config.optionxform = str

    config['Dimensions'] = {'Width (um)': 2000, 'Length (um)': 20}
    config['Transfer'] = {'Preread (ms)': 30000.0,
                          'First Bias (ms)': 120000.0,
                          'Vds (V)': -0.60}

    config['Output'] = {'Preread (ms)': 500.0,
                        'First Bias (ms)': 200.0,
                        'Output Vgs': 4,
                        'Vgs (V) 0': -0.1,
                        'Vgs (V) 1': -0.3,
                        'Vgs (V) 2': -0.5,
                        'Vgs (V) 3': -0.9}
    
    if 'pathlib' in str(type(path)):

        with open(path / 'config.cfg', 'w') as configfile:    
            config.write(configfile)
        return path / 'config.cfg'

    with open(path + r'\config.cfg', 'w') as configfile:
        config.write(configfile)
    return path + r'\config.cfg'
    

def config_file(cfg):
    """
    Generates parameters from supplied config file
    """
    config = configparser.ConfigParser()
    config.read(cfg)
    params = {}
    options = {}

    dim_keys = {'Width (um)': 'W', 'Length (um)': 'L', 'Thickness (nm)': 'd'}
    vgs_keys = ['Preread (ms)', 'First Bias (ms)', 'Vds (V)']
    vds_keys = ['Preread (ms)', 'First Bias (ms)', 'Output Vgs']
    opts_bools = ['Reverse', 'Average']
    opts_str = ['gm_method']
    opts_flt = ['V_low']

    for key in dim_keys:

        if config.has_option('Dimensions', key):
            params[dim_keys[key]] = config.getfloat('Dimensions', key)

    for key in vgs_keys:

        if config.has_option('Transfer', key):
            params[key] = int(config.getfloat('Transfer', key))

    for key in vds_keys:

        if config.has_option('Output', key):
            val = int(config.getfloat('Output', key))

            # to avoid duplicate keys
            if key in params:
                key = 'output_' + key
            params[key] = val

    if 'Output Vgs' in params:

        params['Vgs'] = []
        for i in range(0, params['Output Vgs']):
            nm = 'Vgs (V) ' + str(i)

            val = config.getfloat('Output', nm)
            params['Vgs'].append(val)

    if 'Options' in config.sections():

        for key in opts_bools:

            if config.has_option('Options', key):
                options[key] = config.getboolean('Options', key)

        for key in opts_str:

            if config.has_option('Options', key):
                options[key] = config.get('Options', key)

        for key in opts_flt:

            if config.has_option('Options', key):
                options[key] = config.getfloat('Options', key)

    return params, options
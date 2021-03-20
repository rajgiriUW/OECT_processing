import configparser

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

    with open(path + r'\config.cfg', 'w') as configfile:
        config.write(configfile)

    return path + r'\config.cfg'

import os
import sys
import pytest
import configparser
import numpy as np

sys.path.insert(0, '..')

import oect
from oect.oect_utils.config import make_config, config_file

# most values are hardcoded - be careful if modifying cfg/txt files
# some tests use different subfolders to avoid conflicts

class TestOECT:
    make_config(os.getcwd())

    # set_params
    ############################################################################

    def test_load_pixel(self):
        test_oect = oect.OECT(folder='tests/test_device/01') 
        test_oect.calc_gms()
        test_oect.thresh()
        
    def test_load_device(self):
        test_oect = oect.OECTDevice(path='tests/test_device/full_device', 
                                    options={'plot':[False, False]})
        

    # test that parameters are read from config
    def test_set_params(self):
        test_oect = oect.OECT(folder='tests/test_device/01')  # called in init
        assert (test_oect.params['W'] == 4000.0
                and test_oect.params['L'] == 20.0
                and test_oect.params['d'] == 4e-8
                and test_oect.params['Preread (ms)'] == 20000
                and test_oect.params['First Bias (ms)'] == 120000
                and test_oect.params['Vds (V)'] == 0
                and test_oect.params['output_Preread (ms)'] == 5000
                and test_oect.params['output_First Bias (ms)'] == 200
                and test_oect.params['Output Vgs'] == 2
                and test_oect.params['Vgs'] == [-0.5, -0.8])

    # test that options are read from config
    def test_set_opts(self):
        test_oect = oect.OECT(folder='tests/test_device/options_test')  # called in init
        assert (test_oect.options['Reverse'] == True
                and test_oect.options['Average'] == False
                and test_oect.options['gm_method'] == 'method'
                and test_oect.options['V_low'] == 10)

    # test that additional parameters can be added from constructor
    def test_set_params_add(self):
        test_oect = oect.OECT(folder='tests/test_device/01', params={'test_param1': 100},
                              options={'test_option1': 200})
        assert (test_oect.params['test_param1'] == 100
                and test_oect.options['test_option1'] == 200)

    # test that if options are not in config, all settings are set to default
    def test_set_params_defaults(self):
        test_oect = oect.OECT(folder=os.getcwd())
        test_oect.set_params({}, {}, {}, {})  # try defaults
        assert (test_oect.options['gm_method'] == 'sg'
                and test_oect.options['Reverse'] == True
                and test_oect.options['Average'] == False
                and test_oect.options['V_low'] == False
                and test_oect.options['overwrite'] == False)

    # test that TypeError is raised when parameters passed are not dicts
    def test_set_params_not_dict(self):
        test_oect = oect.OECT(folder=os.getcwd())
        with pytest.raises(IndexError) or pytest.raises(TypeError):
            test_oect.set_params([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])

    # filelist
    #####################################################################

    # test that txt filelist is correctly grabbed from path
    def test_filelist(self):
        test_oect = oect.OECT(folder='tests/test_device/01')
        test_oect.filelist()
        assert (os.path.join('tests/test_device/01', 'uc1_kpf6_output_0.txt') in test_oect.files
                and os.path.join('tests/test_device/01', 'uc1_kpf6_output_1.txt') in test_oect.files
                and os.path.join('tests/test_device/01', 'uc1_kpf6_transfer_0.txt') in test_oect.files
                and test_oect.config[0] == os.path.join('tests/test_device/01', 'uc1_kpf6_config.cfg'))

    # test that config file is generated when folder starts with no cfg
    @pytest.mark.xfail
    def test_filelist_noconfig(self):
        test_oect = oect.OECT(folder='tests/test_device/no_config')
        config_check = 'config.cfg' in os.listdir('tests/test_device/no_config')
        try:
            os.remove('tests/test_device/no_config/config.cfg')
        except:
            pass
        assert config_check

    # get_metadata
    #####################################################################

    # test that metadata is correctly taken from file
    def test_get_metadata(self):
        test_oect = oect.OECT(folder='tests/test_device/01')
        test_file = 'tests/test_device/metadata_test/uc1_kpf6_output_0.txt'
        test_oect.get_metadata(test_file)
        assert (test_oect.Vg == -.5
                and test_oect.W == 2000
                and test_oect.L == 20)

    # test that metadata is correctly grabbed from data file if config doesn't exist
    def test_get_metadata_no_config(self):
        test_oect = oect.OECT(folder='tests/test_device/metadata_test')
        test_oect.make_config = True
        test_file = 'tests/test_device/metadata_test/uc1_kpf6_output_0.txt'
        test_oect.get_metadata(test_file)
        assert (test_oect.Vg == -.5
                and test_oect.W == 2000
                and test_oect.L == 10)

    # transfer_curve
    ######################################################################

    # test that KeyError raised when file not correctly formatted
    def test_transfer_curve_wrong_col_names(self):
        test_oect = oect.OECT(folder=os.getcwd())
        with pytest.raises(KeyError):
            test_file = 'tests/test_device/broken/broken_uc1_kpf6_transfer_0.txt'
            test_oect.get_metadata(test_file)
            test_oect.transfer_curve(test_file)

    # output_curve
    ###################################################################

    # test that KeyError raised when file not correctly formatted
    def test_output_curve_wrong_col_names(self):
        test_oect = oect.OECT(folder=os.getcwd())
        with pytest.raises(KeyError):
            test_file = 'tests/test_device/01/uc1_kpf6_output_0.txt'
            test_oect.get_metadata(test_file)
            test_oect.transfer_curve(test_file)

    # all_outputs
    ###################################################################

    # test that correct number of outputs were added
    def test_all_outputs(self):
        test_oect = oect.OECT(folder='tests/test_device/01')  # called in init
        assert test_oect.num_outputs == 4

    # test that outputs are added to existing outputs
    def test_all_outputs_append(self):
        test_oect = oect.OECT(folder='tests/test_device/01')  # called in init
        test_oect.all_outputs()  # call again
        assert test_oect.num_outputs == 8

    # all_transfers
    ##################################################################

    # test that correct number of transfers were added
    def test_all_transfers(self):
        test_oect = oect.OECT(folder='tests/test_device/01')  # called in init
        assert test_oect.num_transfers == 2

    # _reverse
    ###################################################################

    # test that correct values returned when sweep was not performed
    def test_reverse_no_sweep(self):
        test_oect = oect.OECT(folder=os.getcwd())
        v = np.arange(start=-1, stop=.1, step=.1)
        assert len(v) // 2, False == test_oect._reverse(v)

    # test that correct values returned when sweep was performed
    def test_reverse_with_sweep(self):
        test_oect = oect.OECT(folder=os.getcwd())
        a = np.arange(start=-1, stop=1.1, step=.1)
        b = np.arange(start=.9, stop=-1.1, step=-.1)
        v = np.concatenate((a, b))
        assert len(v) // 2, True == test_oect._reverse(v)

    # update_config
    ######################################################################

    # test that config file is updated to match oect attributes
    def test_update_config(self):
        test_oect = oect.OECT(folder='tests/test_device/no_config')
        # config will be auto generated in init
        # default config will be made with values:
        # [Dimensions]
        # Width (um) = 2000
        # Length (um) = 20

        # [Transfer]
        # Preread (ms) = 30000.0
        # First Bias (ms) = 120000.0
        # Vds (V) = -0.6

        # [Output]
        # Preread (ms) = 500.0
        # First Bias (ms) = 200.0
        # Output Vgs = 4
        # Vgs (V) 0 = -0.1
        # Vgs (V) 1 = -0.3
        # Vgs (V) 2 = -0.5
        # Vgs (V) 3 = -0.9
        test_oect.W = 4000
        test_oect.L = 10
        test_oect.Vd = -1
        test_oect.Vg_array = [0, .1, .2, .3, .4]
        test_oect.update_config()
        config = configparser.ConfigParser()
        config.read(test_oect.config)
        assert (config['Dimensions']['Width (um)'] == '4000'
                and config['Dimensions']['Length (um)'] == '10'
                and config['Transfer']['Vds (V)'] == '-1'
                and config['Output']['Preread (ms)'] == '500.0'
                and config['Output']['First Bias (ms)'] == '200.0'
                and config['Output']['Vgs (V) 0'] == '0'
                and config['Output']['Vgs (V) 1'] == '0.1'
                and config['Output']['Vgs (V) 2'] == '0.2'
                and config['Output']['Vgs (V) 3'] == '0.3'
                and config['Output']['Vgs (V) 4'] == '0.4')
        try:
            os.remove('tests/test_device/no_config/config.cfg')
        except:
            pass

    # make_config
    ######################################################################

    # test that FileNotFoundError thrown when provided with invalid path
    @pytest.mark.xfail
    def test_make_config_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            make_config('a_nonexistent_path')

    # config_file
    #############################################################

    # test that params exist when loaded from config
    def test_config_file_params(self):
        params, opts = config_file('tests/config.cfg')
        assert bool(params) and not bool(opts)

    # test that options exist when loaded from config
    def test_config_file_opts(self):
        params, opts = config_file('tests/test_device/options_test/uc1_kpf6_config.cfg')
        assert bool(params) and bool(opts)

    # tests that nothing is added when provided with invalid path
    def test_config_file_invalid_file(self):
        params, opts = config_file('a_nonexistent_file')
        assert not bool(params)

    # test that configparser error thrown when provided with non-cfg file
    def test_config_not_cfg(self):
        with pytest.raises(configparser.MissingSectionHeaderError):
            params, opts = config_file('tests/dummy_file.py')

# questions/why I didn't write tests for these functions
# loaddata - just consolidates a lot of functions
# calc_gms - what would an expected gm would be?
# _calc_gm - ""
# quadrant - what does the self.gm_peaks.index indicate? what
#	would be an expected value?
# thresh - confused about the zip calls, gm_peaks
# _min_fit - what would an expected result be?
# line_f - self-explanatory enough to not test
# _find_peak - what would an expected result be?

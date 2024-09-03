import warnings

import numpy as np
from scipy import signal as sps
from scipy import interpolate as spi


"""
Derivative for generating gm from Id-Vg plots
"""


def gm_deriv(v, i, method='raw', fit_params={'window': 11, 
                                             'polyorder': 2, 
                                             'deg': 8,
                                             'k': 5}):
    '''
    :param v:
	:type v:
	
    :param i:
    :type i:
	
	:param method:
    :type method: str
	
	:param fit_params:
    :type fit_params: dict
	
    :returns:
	:rtype:
    '''
    if method == 'sg':
        # Savitsky-Golay method
        if not fit_params['window'] & 1:  # is odd
            fit_params['window'] += 1
        gml = sps.savgol_filter(i.T, window_length=fit_params['window'],
                                polyorder=fit_params['polyorder'], deriv=1,
                                delta=v[2] - v[1])
        if len(gml.shape) > 1:
            gml = gml[0, :]

    elif method == 'raw':
        # raw derivative
        gml = np.gradient(i.flatten(), v[2] - v[1])

    elif method == 'poly':
        # polynomial fit
        funclo = np.polyfit(v, i, fit_params['deg'])
        gml = np.gradient(np.polyval(funclo, v), (v[2] - v[1]))

    else:
        warnings.warn('Bad gm_method, aborting')
        return

    return gml

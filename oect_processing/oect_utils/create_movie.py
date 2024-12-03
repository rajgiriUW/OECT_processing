import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt
import sys
import tifffile as tf

'''
Note: A 'str is not callable' bug is often due to not running set_mpeg

'''


def set_mpeg(path=None):
    '''

    :param path:
    :type path: str
    '''
    if not path:
        plt.rcParams[
            'animation.ffmpeg_path'] = r'C:/Users/gingerlabsp/Downloads/ffmpeg/bin/ffmpeg.exe'

    else:
        plt.rcParams['animation.ffmpeg_path'] = path
    return


def create_freq_movie(imgs,
                      filename='inst_freq',
                      smooth=None,
                      size=(10, 6),
                      cmap='inferno',
                      interval=60,
                      repeat_delay=100,
                      crop=None,
                      verbose = True):
    '''
    Creates an animation that goes through all the instantaneous frequency data.

    :param h5_ds: The instantaneous frequency data. NOT deflection, this is for post-processed data
    :type h5_ds: USID dataset

    :param filename:
    :type filename: str

    :param time_step:
        10 @ 10 MHz = 1 us
        50 @ 10 MHz = 5 us
    :type time_step: int, optional

    :param idx_start: What index to start at. Typically to avoid the Hilbert Transform edge artifacts, you start a little ahead
    :type idx_start: int

    :param idx_stop: Same as the above,in terms of how many points BEFORE the end to stop
    :type idx_stop: int

    :param smooth: Whether to apply a simple boxcar smoothing kernel to the data
    :type smooth: int, optional

    :param size: figure size
    :type size: tuple, optional

    :param vscale: To hard-code the color scale, otherwise these are automatically generated
    :type vscale: list [float, float], optional

    :param cmap:
    :type cmap: str, optional

    :param interval: The delay time in milliseconds. 1000/fps 
    :type interval:

    :param repeat_delay: Used when saving to set the delay for when the mp4 repeats from the start
    :type repeat_delay: int

    :param crop: Crops the image to a certain line, in case part of the scan is bad
    :type crop: int

    :param verbose: Whether to print info to the console about the progress
    :param discharge_time: Whether to change the title based on discharging data
    :param discharge: index at which to change the title

    '''

    set_mpeg()
    fig, ax = plt.subplots(figsize = size)

    # Loop through time segments
    ims = []
       
    for imgfile in imgs:
        img = tf.imread(imgfile)
        for n in range(img.shape[0]):
            im0 = ax.imshow(img[n, :, :], cmap=cmap, origin='lower', animated=True)
            ims.append([im0])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat_delay=repeat_delay)

    try:
        ani.save(filename + '.mp4')
    except TypeError as e:
        print(e)
        print('A "str is not callable" message is often due to not running set_mpeg function')

    return

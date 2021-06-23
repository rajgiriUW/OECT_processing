# Organic Electrochemical Transistor (OECT) device processing

This package processes sets of OECT device measurements in Python via Pandas, with primary users being the Ginger Lab at University of Washington.
It includes some simplified extraction and processing of recorded data.  
There is also simplified scripts for extracting and processing spectroelectrochemistry data.  

All pixels are expected to have a .cfg file or will auto-generate one. These files contain:
* Width and Length
* Film thickness
* Vg and Vd for output and transfer curves
* Read times (dwell time before first measurement and before each subsequent voltage step)

An example can be found in the ```tests/test_device/01/uc1_kpf6_config.cfg``` file.

## Installation/Using from Source
To install, download the source code locally and then run in some command prompt (for many users, this would be running "Anaconda Prompt" then navigating to the directory you saved this code to):

```python setup.py```

To edit locally:

```python setup.py develop```


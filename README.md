# Organic Electrochemical Transistor (OECT) device processing

```
Contact info: 

Rajiv Giridharagopal, Ph.D.
University of Washington
Department of Chemistry
rgiri@uw.edu
```

This package processes sets of OECT device measurements in Python via Pandas, with primary users being the Ginger Lab at University of Washington.
It includes some simplified extraction and processing of recorded data.  
There is also simplified scripts for extracting and processing spectroelectrochemistry data. 
Included is a simple **Streamlit app**, see below.

All pixels are expected to have a .cfg file or will auto-generate one. These files contain:
* Width and Length
* Film thickness
* Vg and Vd for output and transfer curves
* Read times (dwell time before first measurement and before each subsequent voltage step)

An example can be found in the ```tests/test_device/01/uc1_kpf6_config.cfg``` file.

## Installation/Using from Source
To install, download the source code locally and then run in some command prompt (for many users, this would be running "Anaconda Prompt" then navigating to the directory you saved this code to):

```python setup.py install```

To edit locally:

```python setup.py develop```

## Instructions
### Option 1: Jupyter Notebook

The easiest way to run this is by loading Jupyter notebook and going through the [OECT Processing Notebook](/oect_processing/notebooks/OECT%20Processing.ipynb)

### Option 2: Streamlit App

There is also a fairly simple Streamlit app to use this. To open this:

1) Open a command prompt (e.g. "Anaconda Prompt") and navigate to this repository's directory

2) Type ```streamlit run oect_app.py```

3) The app should open on your browser. Use the instructions in the **GUIDE** dropdown box. Basically, just copy and paste a path to your local data on the left panel. Then, select the devices to use for calculating uC*

#### Initial screen if the App loads
![Initial Loading](/imgs/app_screenshot_1.png?raw=true)

#### Instructions, Part 1
![Instructions](/imgs/app_instructions_1.png?raw=true)

#### Instructions, Part 2
![Instructions](/imgs/app_instructions_2.png?raw=true)




# Organic Electrochemical Transistor (OECT) device processing

This small set of files processes sets of OECT device measurements in Python via Pandas, with primary users being the Ginger Lab at University of Washington.
It includes some simplified extraction and processing of recorded data.  
There is also simplified scripts for extracting and processing spectroelectrochemistry data.  

There are two primary modes of operation:  
1. Individual device pixels.  
2. A folder containing pixels for generating a uC* plot. 

All pixels are expected to have a .cfg file or will auto-generate one. These files contain:
* Width and Length
* Film thickness
* Vg and Vd for output and transfer curves
* Read times (dwell time before first measurement and before each subsequent voltage step)

#### Usage (in Spyder or equivalent):
```
>> path = r'\path\to\data_folder'
>> import OECT, OECT_plotting, OECT_loading
>> devices, uC = OECT_loading.uC_scale(path_uC, thickness=100e-9)

>> path = r'\path\to\specific\device'
>> dv = OECT.OECT(path, thickness=100e-9)
>> dv.calc_gms()
>> dv.thresh()
>> OECT_plotting.plot_transfers_gm(dv)
```

This will then generate uC* graph, an averaged transfer curve and output curve, and individual transfer and output curves. Thickness is the film thickness. You will have to know this for these data to be correct, of course, but it is more critical to uC* scaling




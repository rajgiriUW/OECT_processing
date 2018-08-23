# OECT_processing

#### For processing OECT data, assumes data are saved in this sort of folder structure:

.../avg/01, /02, /03, /04
              
.../uC/01, /02, /03, /04, /05 

(05 being a copy of any of the avg pixels)

Where you set variables:
  path_avg = r'.../avg'
  path_uC = r'.../uC' 

and replace '...' with the actual system path

#### Usage (in Spyder or equivalent):
```
>> import OECT, OECT_plotting, OECT_loading
>> path_avg = r'path_to_avg_folder' 
>> pixels, Id_Vg, Id_Vd = OECT_loading.load_avg(path_avg, thickness=100e-9, plot=True)
>> pixels, Wd_L, gms = OECT_loading.uC_scale(path_uC, thickness=100e-9, plot=True
```

#### This will then generate uC* graph, an averaged transfer curve and output curve, and individual transfer and output curves. Thickness is the film thickness. You will have to know this for these data to be correct, of course, but it is more critical to uC* scaling




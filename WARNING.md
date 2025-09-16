# Warning
In my running of this codebase, I was unable to make the codebase work with newer GPUs even after loading cuda/11.0 versions. This is a bug with the latest numba package supported by Python 3.6. 

Rather than updating the entire codebase, a workaround can be seen by going to:
`lib_path/python3.6/site-packages/numba/cuda/cudadrv/devices.py`

After line 155, manually add the following:
`ac.devnum = 0`

This will overwrite the incorrectly identified GPU number to whatever device 0 is (which was my GPU). This worked for me, but there is no guarantee it is entirely reproducable.
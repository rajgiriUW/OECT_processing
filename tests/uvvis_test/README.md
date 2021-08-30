UVVIS TESTING OVERVIEW
=======================
Testing uvvis.py requires 3 core files:
* **uvvis_inputs.py** contains the parameters for generating expected values and testing uvvis functions. This exists so you don't need to worry about hard-coding the same stuff into **generate_uvvis_expected.py** and **uvvis_test.py.**
* **generate_uvvis_expected.py** must be run to generate expected values of uvvis functions. The outputs will depend on parameters imported from **uvvis_inputs.py.** 
* **uvvis_test.py** contains the actual tests that pytest will run. The tests will use parameters imported from **uvvis_inputs.py.** Tests outputs are compared to the generated expected values.

Thus, a testing cycle will go something like this:
1. Start with a uvvis.py that you know is functional, so it will generate the correct expected values.
2. If needed, edit **uvvis_inputs.py** to hold the parameters you would like to use for testing.
3. Run **generate_uvvis_expected.py** to create expected values based on parameters from **uvvis_inputs.py.** The resulting files will be saved into a subfolder.
4. Run pytest on **uvvis_test.py.**
5. Based on pytest results, edit and debug uvvis.py.
6. Repeat steps 4-5 until all tests pass.

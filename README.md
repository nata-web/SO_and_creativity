# Untapped Potential in Self-Optimization of Hopfield Networks: The Creativity of Unsupervised Learning
This repository contains the code used for the paper <b>"Untapped Potential in Self-Optimization of Hopfield Networks: The Creativity of Unsupervised Learning"</b> ([arXiv](https://arxiv.org/abs/2501.04007)). 

## Content:

* **SO_base.py** - Main file. Uses the custom containers in *containers.py*, the parameters in *paramfiles/modularprob.py*, and calls for *SO_base_FUNC.py* to run the simulations. Calls for *SO_base_plots.py* to generate figures 2-9 in the paper. 
* **SO_base_plots.py** - Calls for *SO_for_SAT.py* and *borders.py*, and contains various plot functions to generate figures 2-9.
* **SO_base_FUNC.py** - Calls for *hebbclean.F90* to run the simulation in FORTRAN. Contains functions from [SO-scaled-up](https://github.com/nata-web/SO-scaled-up/tree/main) to generate random weight matrices.
* **hebbclean.F90** - Contains the FORTRAN routine.
* **containers.py** - Contains custom containers.
* **paramfiles/modularprob.py**  - Folder that contains all the parameters.

To replicate the figures in the paper, modify the following parameters in the *paramfiles/modularprob.py* file:
* For Figures 2-4, and 8-9:
  * Set ```CO.alphaGrid = True``` and  ```CO.num_alphas = 1``` (for Fig. 3 also set ```plot6 = True``` in *SO_base.py*).
* For Figures 5-6:
  * Set ```CO.alphaGrid = False``` and  ```CO.num_alphas = 64```.
* For Figure 7:
  * Set ```CO.alphaGrid = False```,  ```CO.num_alphas = 1``` and ```CO.resets_arr = [4000]  * len(CO.alpha_arr)``` in line 33 of the code.

## To run the code from Python with FORTRAN:
Make sure your system has gfortran and f2py. Run the following commands before the execution of the python code to compile the FORTRAN file:

`f2py3 --f90flags="-g -fdefault-integer-8 -O3" -m hebbF -c hebbclean.F90`

## To run the simulations from Python without FORTRAN
Current code is written to run the explicitly with FORTRAN (plots are generated in Python). To run the simulation in Python only see [SO-scaled-up](https://github.com/nata-web/SO-scaled-up/tree/main).

## 

If you have any questions, feel free to open an issue or send me an email: natalya.weber (at) oist.jp.

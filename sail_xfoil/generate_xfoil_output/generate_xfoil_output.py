# %% Imports

import os
import subprocess
import numpy

# %% Inputs

# calling "NACA xxxx" maps to specific airfoil design parameters - can later be replaced by own airfoil designs
airfoil_name = "NACA 1234"
# initial/final angle of attack & evaluation steplength
alpha_i = 0
alpha_f = 10
alpha_step = 0.25
# viscosity modifier
Re = 100000
n_iter = 100

# %% xfoil file writer

if os.path.exists("output_file.txt"):
    os.remove("output_file.txt")

input_file = open("input_file.in", 'w')
input_file.write("LOAD {0}.dat\n".format(airfoil_name))
input_file.write(airfoil_name + '\n')
input_file.write("PANE\n")
input_file.write("OPER\n")
input_file.write("Visc {0}\n".format(Re))
input_file.write("PACC\n")
input_file.write("output_file.txt\n\n")
input_file.write("ITER {0}\n".format(n_iter))
input_file.write("ASeq {0} {1} {2}\n".format(alpha_i,
                                             alpha_f,
                                             alpha_step))
input_file.write("\n\n")
input_file.write("quit\n")
input_file.close()

xfoil_path = r"/mnt/c/'Program Files'/xfoil/./xfoil.exe"
subprocess.call(xfoil_path + " < input_file.in", shell=True)
output_data = numpy.loadtxt("output_file.txt", skiprows=12)
# %%

# MonteCarlo
This is a monte carlo implementation with cuda. We use the alogrithm described in investopedia: https://www.investopedia.com/terms/m/montecarlosimulation.asp

To make the binary files simply type the following into terminal:

$ make

The code works with any yahoo data we have provided an example of Nvidia to run the code with the Nvidia run the following command in terminal:

$ ./vector NVDA.csv mt1


notice the mt1 option you have many options for different random number generators.

mt1 = mersenne twister 1

mt2 = mersenne twister 2

mrg = multiple recursive generator AKA linear congruential

phi = philox

sobol = SOBOL

make sure to specify at least 1 or the code will segfault.

Once finished the code will generate a folder named output and a file named random.txt

You can also change BLOCK_SIZE, SIM_SIZE,and VSIZE in the vector.h file.

VSIZE corresponds to the amount of data in the .csv make sure that vsize does not exceed this value.
SIM_SIZE is the amount of simulations that the code generates.

-----------------------------------------------------PLOTS----------------------------------------------------

The plotting python scripts have dependecy on pandas, matplotlib,and numpy all in python3.
You will not be able to use these in the gpulabs machines at the University of Minnesota. 
To use the scripts copy the output folder and .py files to an enviornment with the graphs above, 
and run the code.

to run the .py files simply type the following in terminal:

$ python3 <name_of_python_file>.py

Then sit back and watch the pretty graphs.

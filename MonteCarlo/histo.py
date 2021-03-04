import numpy as NP
from matplotlib import pyplot as PLT


with open('./random.txt') as f:
  v = NP.loadtxt(f, delimiter=",\n", dtype='float', comments="#", skiprows=1, usecols=None)
  PLT.hist(v,bins=100)
  PLT.ylabel("Frequency")
  PLT.xlabel("Random Vector")
  PLT.show()

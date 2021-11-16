import Agent
from Agent import *
from readin import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
 Aim : Taking a pair of neurons (example 0 and 5); check how correlated are their result in a specific world state.
 Rho : Has One scalar value for correlation.
 Take multivariate normal function from python and draw variables, draw mean and covariance.
 
 Define the distribution by mean and covariance.  ---> Tuning Curve.
 Sample from multivariate normal distribution and then plot. X Axis : value from 1st neuron and Y Axis : value from 2nd Neuron.
 
 If they are continuous then we will get a good function. ( sample multiple times)
 
 Mean, Covariance, size Sample Back --->> LOOP (plot this in X)
 
 Sample from p(x|w) 
 
 Trying to import variables from the Agent class in Agent.py in order to identify and display values pertaining to the bivariate distribution"""

plt.scatter(sample_B[:][0], sample_B[:][1], edgecolors="black", color="darkred", label="Neuron 0/1")
plt.scatter(sample_B[:][0], sample_B[:][2], edgecolors="black", color="firebrick", label="Neuron 0/2")
plt.scatter(sample_B[:][0], sample_B[:][3], edgecolors="black", color="red", label="Neuron 0/3")
plt.scatter(sample_B[:][0], sample_B[:][4], edgecolors="black", color="crimson", label="Neuron 0/4")
plt.scatter(sample_B[:][0], sample_B[:][5], edgecolors="black", color="magenta", label="Neuron 0/5")
plt.scatter(sample_B[:][0], sample_B[:][6], edgecolors="black", color="purple", label="Neuron 0/6")
plt.scatter(sample_B[:][0], sample_B[:][7], edgecolors="black", color="violet", label="Neuron 0/7")


plt.scatter(sample_B[:][0], sample_B[:][1], edgecolors="black", color="violet", label="Neuron 0/1")
plt.scatter(sample_B[:][0], sample_B[:][2], edgecolors="black", color="indigo", label="Neuron 0/2")
plt.scatter(sample_B[:][0], sample_B[:][3], edgecolors="black", color="blue", label="Neuron 0/3")
plt.scatter(sample_B[:][0], sample_B[:][4], edgecolors="black", color="green", label="Neuron 0/4")
plt.scatter(sample_B[:][0], sample_B[:][5], edgecolors="black", color="yellow", label="Neuron 0/5")
plt.scatter(sample_B[:][0], sample_B[:][6], edgecolors="black", color="orange", label="Neuron 0/6")
plt.scatter(sample_B[:][0], sample_B[:][7], edgecolors="black", color="red", label="Neuron 0/7")

# plt.scatter(listbyruns[1, :, 1, 0], listbyruns[1, :, 1, 1], edgecolors="black", color='red', label="W1")
# plt.scatter(listbyruns[1, :, 2, 0], listbyruns[1, :, 2, 1], edgecolors="black", color='green', label="W2")
# plt.scatter(listbyruns[1, :, 3, 0], listbyruns[1, :, 3, 1], edgecolors="black", color='violet', label="W3")
# plt.scatter(listbyruns[1, :, 4, 0], listbyruns[1, :, 4, 1], edgecolors="black", color='indigo', label="W4")
# plt.scatter(listbyruns[1, :, 5, 0], listbyruns[1, :, 5, 1], edgecolors="black", color='cyan', label="W5")
# plt.scatter(listbyruns[1, :, 6, 0], listbyruns[1, :, 6, 1], edgecolors="black", color='brown', label="W6")
# plt.scatter(listbyruns[1, :, 7, 0], listbyruns[1, :, 7, 1], edgecolors="black", color='orange', label="W7")
# #plt.scatter(listbyruns[1, 0, :, 0], listbyruns[1, 7, :, 1], edgecolors="black", color='yellow', label="W8")
# plt.legend()
# plt.title("Multivariate Normal - Run 1, N(all) vs N1 to 7 W0 - W1?")
# plt.show(False)


# try to write a function for plotting multivariate normals:

# 3 variables have to be constant in each loop
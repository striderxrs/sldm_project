import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#put in results folder

# extract variables from last episode (last update step)
def extract(path, file_names):
    I_W_A = []
    eus = []
    rhos = []
    x_means = []
    I_X_A = []
    betas = []
    pxw_means = []
    pxw_sigmas = []
    for file in range(len(file_names)):
        #print("load file number:", file)
        with open(os.path.join(path, file_names[file]), 'rb') as handle:
            storage = pickle.load(handle)
            x_means.append([int(x) for x in storage.x_means])
            I_W_A.append(storage.dkl_w_e[-1])
            I_X_A.append(storage.I_X_A_e[-1])
            eus.append(storage.exp_u_e[-1])
            rhos.append(storage.rhos_e[-1])
            betas.append(storage.betas)
            pxw_means.append(storage.pxw_means_e[-1])
            pxw_sigmas.append(storage.pxw_sigmas_e[-1])

    x_means = np.array(x_means)
    I_W_A = np.array(I_W_A)
    I_X_A = np.array(I_X_A)
    eus = np.array(eus)
    rhos = np.array(rhos)
    betas = np.array(betas)
    pxw_sigmas = np.array(pxw_sigmas)
    pxw_means = np.array(pxw_means)
    return x_means, betas, I_W_A, I_X_A, rhos, eus, pxw_sigmas, pxw_means


# extract rhos from all episodes (full update process) e is episode
def extract_rhos_e(path, file_names):
    rhos = []
    for file in range(len(file_names)):
        print("load file number:", file)
        with open(os.path.join(path, file_names[file]), 'rb') as handle:
            storage = pickle.load(handle)
            rhos.append(storage.rhos_e)
    rhos = np.array(rhos)
    return rhos


# EXAMPLE
path = os.path.join('F:/SerialCase_NoiseCorrelationGD/results', 'test')  # path to subdirectory 'test' in result-folder (=foldername in "serial_gradient.py")
file_names = [f for f in os.listdir(path)]

x_means, betas, I_W_A, I_X_A, rhos, eus, pxw_sigmas, pxw_means = extract(path, file_names)
rhos_e = extract_rhos_e(path, file_names)

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


w = 0 #worldstate (loop)
x_size = np.size(x_means, 1)
print("xsize = ", x_size)
rhos_test = rhos[8, :, :]
sigmas_test = pxw_sigmas[8, :, :]


def covariance(rhos_9 ,w ,sigmas_9):
    cov = np.einsum('i,j,ij->ij', sigmas_test[w], sigmas_test[w], rhos_test[w] + np.eye(x_size))
    #print(pxw_sigmas[w])
    #print(np.size(pxw_sigmas[w]))
    return cov

cov_test = covariance(rhos_test, w, sigmas_test)
#cov8 = covariance(rhos[7, :, :], w)
#print(cov_test)
#print(np.shape(cov_test))
#print(np.shape(pxw_means))

#################################################################################################################
print(" I HAVE REACHED HERE ")
# collecting covariances of each run

# plt.scatter([0.0, 0.1, 0.2], [0.0, 1.0, 0.2])
# plt.show()

exact_runs = np.shape(rhos)
exact_runs = exact_runs[0]

print("len rhos 0", len(rhos[0]))
print("exact_runs", exact_runs)

number_runs = []
for i in range(exact_runs):
    number_runs.append(i)
print("number_runs = %s" % number_runs)

world_states = []
for i in range(len(rhos[1])):
    world_states.append(i)
print("world_states = %s" % world_states)

print("rhos shape =", np.shape(rhos))
print("rhos length =", np.shape(rhos[1]))
print("pxw means shape", np.shape(pxw_means))

from scipy.stats import multivariate_normal

# Covariance Matrix generation - Extract Covariance values between rhos and pxw sigmas in the same run
# Idea = x is range of runs, y is world states. Get covariance matrix of each x and y through a loop
for x in number_runs:
    for y in world_states:
        cov_matrices = covariance(rhos[x, :, :], y, pxw_sigmas[x, :, :])
        print(np.shape(cov_matrices))
        print(cov_matrices)
        print("END OF LOOP %d = ", x)

mus_custom = [0.8,0.57,0.47,0.43,0.4,0.43,0.47,0.57]


x = number_runs #[0 - 8]
y = world_states #[0 - 15]
n = 0
for i in x:
    sample_B = []
    for j in y:
        sample_A = multivariate_normal.rvs(mean=pxw_means[x[i], y[j]], cov=cov_matrices[x[i], y[j]], size=100)
        # sample_A = multivariate_normal.rvs(mean=range(8), cov=cov_matrices[x[i], y[j]], size=100)
        print("are we here?")
        #print(sample_A[:, 2])
        #print("----------")
        #print(sample_A[:, 7])
        #print("Sample A", sample_A)
        #print("dimensions", np.shape(sample_A))
        sample_B.append(sample_A)
        #plt.scatter(sample_A[:, 0], sample_[:, 1])
    sample_B = np.array(sample_B)
    print(np.shape(sample_B))

    plt.scatter(sample_B[:, 0], sample_B[:, 1])
plt.show()





import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
from matplotlib import cm
from pylab import rcParams
import sys

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
        # print("load file number:", file)
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
path = os.path.join('F:/SerialCase_NoiseCorrelationGD/results',
                    'test')  # path to subdirectory 'test' in result-folder (=foldername in "serial_gradient.py")
file_names = [f for f in os.listdir(path)]

x_means, betas, I_W_A, I_X_A, rhos, eus, pxw_sigmas, pxw_means = extract(path, file_names)
rhos_e = extract_rhos_e(path, file_names)

x_size = 8

############################################################################################

def covariance(rhos, sigmas, w):
    cov = np.einsum('i,j,ij->ij', sigmas[w], sigmas[w], rhos[w] + np.eye(x_size))
    return cov

print('pxw Sigmas', pxw_sigmas)
print('pxw means', pxw_means)

# data retrieval (query information using number_runs)


number_runs = range(np.size(rhos, 0))
world_states = range(np.size(rhos, 1))
print('shape of pxw means: ', np.shape(pxw_means))
print('pxw-means run 2:', pxw_means[2,:,:])
print('pxw-means run 10:', pxw_means[10,:,:])
print('rhos shape', number_runs)
avgn = np.mean(rhos[2, :, 0, 1:7])
print(avgn)
print(np.shape(rhos[2, :, 0, 1:7]))


#sys.exit(0)

# get covariances
cov_matrices = []
for x in number_runs:
    cov_matrices_run = []
    for y in world_states:
        # print(np.shape(rhos[x]))
        # print(np.shape(pxw_sigmas[x]))
        cov_matrices_run.append(covariance(rhos[x, :, :], pxw_sigmas[x, :, :], y))
    cov_matrices.append(np.array(cov_matrices_run))
cov_matrices = np.array(cov_matrices)
print(np.shape(cov_matrices))

# sample from pxw
listbyruns = []
for x in number_runs:
    sample_B = []  # samples for all worldstates
    std_dev = []
    nmean = []
    for y in world_states:
        sample_A = multivariate_normal.rvs(mean=pxw_means[x, y, :], cov=cov_matrices[x, y, :, :],
                                           size=300)  # samples from single w
        std_dev_values = np.std(sample_A)
        nmean_values = np.mean(sample_A)
        sample_B.append(sample_A)
        nmean.append(nmean_values)
        std_dev.append(std_dev_values)
    listbyruns.append(np.array(sample_B))
    print("listbyruns", np.shape(listbyruns))

print(np.shape(listbyruns))
sample_B = np.array(sample_B)
listbyruns = np.array(listbyruns)
print(np.shape(sample_B))
print("standard deviation", std_dev)
print("mean", nmean)
print("Rho vector shape : ", np.shape(rhos))

rcParams['figure.figsize'] = 13, 13
plt.plot(pxw_means[10, :, 0], label='mus1 = [0.80, 0.57, 0.47, 0.43, 0.40, 0.43, 0.47, 0.57]')
plt.plot(pxw_means[11, :, 0], label='mus2 = [0.10, 0.12, 0.20, 0.40, 0.20, 0.12, 0.10, 0.05]')
plt.plot(pxw_means[12, :, 0], label='mus3 = [0.90, 0.40, 0.50, 0.68, 0.78, 0.68, 0.50, 0.40]')
plt.plot(pxw_means[13, :, 0], label='mus4 = [0.98, 0.80, 0.75, 0.70, 0.65, 0.70, 0.75, 0.80]')
plt.plot(pxw_means[14, :, 0], label='mus5 = [0.95, 0.45, 0.38, 0.30, 0.22, 0.30, 0.38, 0.45]')
plt.xlabel('World States')
plt.title('Tuning Curve values for each World State[N0]')
plt.legend()
plt.show()

#ERRORBAR NEURON 0 vs 1:7
npmean = np.mean(rhos[:, :, 0, 1:7], axis=2)  # averaging over the RHO values of all pairs of neurons with N0
meanoverruns = np.mean(npmean, axis=0)
print(np.size(meanoverruns))
npstd = np.std(npmean, axis=0)  # standard error of mean formula

sem = npstd / (np.sqrt(len(npmean)))
xs = range(8)

#plotting rhos. look for different mus values
rcParams['figure.figsize'] = 15, 15
fig, rhoplot = plt.subplots(3, 2)
plt.suptitle("Comparing RHO values for different Tuning Curves. N0 - N1:7", size=15)

local_means = np.mean(rhos[10:14, :, 0, 1:7], axis=2)
local_mean = np.mean(local_means, axis=0)
local_std = np.std(local_means, axis=0)
local_sem = local_std/np.sqrt(len(local_means))

avg10 = np.mean(rhos[10, :, 0, 1:7], axis=1)
avg10std = np.std(rhos[10, :, 0, 1:7], axis=1)
rhoplot[0][0].plot(avg10, color='black')
rhoplot[0][0].errorbar(xs, avg10, avg10std)
rhoplot[0][0].errorbar(xs, local_mean, local_sem)
rhoplot[0][0].set_title("Run 10 (mus1)[0.80,0.57,0.47,0.43,0.40,0.43,0.47,0.57]", size=9)
rhoplot[0, 1].scatter(listbyruns[10, 0:7, :, 0], listbyruns[10, 0:7, :, 1], edgecolors="black", label='Run 10[mus1]')
rhoplot[0, 1].legend()

avg11 = np.mean(rhos[11, :, 0, 1:7], axis=1)
avg11std = np.std(rhos[11, :, 0, 1:7], axis=1)
rhoplot[1][0].plot(avg11, color='black')
rhoplot[1][0].errorbar(xs, avg11, avg11std)
#rhoplot[1][0].errorbar(xs, meanoverruns, sem)
rhoplot[1][0].set_title("Run 11 (mus2)[0.10, 0.12, 0.20, 0.40, 0.20, 0.12, 0.10, 0.05]", size=9)
rhoplot[1, 1].scatter(listbyruns[11, 0:7, :, 0], listbyruns[11, 0:7, :, 1], edgecolors="black", label='Run 11[mus2]')
rhoplot[1, 1].legend()

avg12 = np.mean(rhos[12, :, 0, 1:7], axis=1)
avg12std = np.std(rhos[12, :, 0, 1:7], axis=1)
rhoplot[2][0].plot(avg12, color='black')
rhoplot[2][0].errorbar(xs, avg12, avg12std)
#rhoplot[2][0].errorbar(xs, meanoverruns, sem)
rhoplot[2][0].set_title("Run 12 (mus3)[0.90, 0.40, 0.50, 0.68, 0.78, 0.68, 0.50, 0.40]", size=9)
rhoplot[2, 1].scatter(listbyruns[12, 0:7, :, 0], listbyruns[12, 0:7, :, 1], edgecolors="black", label='Run 12[mus3]')
rhoplot[2, 1].legend()

plt.legend()
plt.show()


fig, rhoplot = plt.subplots(2, 2)
rcParams['figure.figsize'] = 15, 15
plt.suptitle("Comparing RHO values for different Tuning Curves(part 2). N0 - N1:7", size=15)

avg13 = np.mean(rhos[13, :, 0, 1:7], axis=1)
avg13std = np.std(rhos[13, :, 0, 1:7], axis=1)
rhoplot[0][0].plot(avg13, color='black')
rhoplot[0][0].errorbar(xs, avg13, avg13std)
#rhoplot[0][0].errorbar(xs, meanoverruns, sem)
rhoplot[0][0].set_title("Run 13 (mus4)[0.98, 0.80, 0.75, 0.70, 0.65, 0.70, 0.75, 0.80]", size=9)
rhoplot[0, 1].scatter(listbyruns[13, 0:7, :, 0], listbyruns[13, 0:7, :, 1], edgecolors="black", label='Run 13[mus4]')
rhoplot[0, 1].legend()

avg14 = np.mean(rhos[14, :, 0, 1:7], axis=1)
avg14std = np.std(rhos[14, :, 0, 1:7], axis=1)
rhoplot[1][0].plot(avg14, color='black')
rhoplot[1][0].errorbar(xs, avg14, avg14std)
#rhoplot[1][0].errorbar(xs, meanoverruns, sem)
rhoplot[1][0].set_title("Run 14 (mus5)[0.95, 0.45, 0.38, 0.30, 0.22, 0.30, 0.38, 0.45]", size=9)
rhoplot[1, 1].scatter(listbyruns[14, 0:7, :, 0], listbyruns[14, 0:7, :, 1], edgecolors="black", label='Run 14[mus5]')
rhoplot[1, 1].legend()

plt.legend()
plt.show()



#plotting rhos. comparing different beta values
rcParams['figure.figsize'] = 15, 15
fig, rhoplot = plt.subplots(5, 2)
plt.suptitle("Comparing RHO values for different Beta Values[pxw_betas] Sigma is 0.02 [Mus1]. N0 - N1:7", size=15)

avg0 = np.mean(rhos[0, :, 0, 1:7], axis=1)
avg0std = np.std(rhos[0, :, 0, 1:7], axis=1)
rhoplot[0, 0].plot(avg0, color='black')
rhoplot[0, 0].errorbar(xs, avg0, avg0std)
#rhoplot[0, 0].errorbar(xs, meanoverruns, sem)
rhoplot[0, 0].set_title("Run 0 : beta 10", size=9)
rhoplot[0, 1].scatter(listbyruns[0, 0:7, :, 0], listbyruns[0, 0:7, :, 1], edgecolors="black", label='Run 0')
rhoplot[0, 1].legend()

avg1 = np.mean(rhos[1, :, 0, 1:7], axis=1)
avg1std = np.std(rhos[1, :, 0, 1:7], axis=1)
rhoplot[1, 0].plot(avg1, color='black')
rhoplot[1, 0].errorbar(xs, avg1, avg1std)
#rhoplot[1, 0].errorbar(xs, meanoverruns, sem)
rhoplot[1, 0].set_title("Run 1 : beta 11", size=9)
rhoplot[1, 1].scatter(listbyruns[1, 0:7, :, 0], listbyruns[1, 0:7, :, 1], edgecolors="black", label='Run 1')
rhoplot[1, 1].legend()

avg2 = np.mean(rhos[2, :, 0, 1:7], axis=1)
avg2std = np.std(rhos[2, :, 0, 1:7], axis=1)
rhoplot[2, 0].plot(avg2, color='black')
rhoplot[2, 0].errorbar(xs, avg2, avg2std)
#rhoplot[2, 0].errorbar(xs, meanoverruns, sem)
rhoplot[2, 0].set_title("Run 2 : beta 12", size=9)
rhoplot[2, 1].scatter(listbyruns[2, 0:7, :, 0], listbyruns[2, 0:7, :, 1], edgecolors="black", label='Run 2')
rhoplot[2, 1].legend()

avg3 = np.mean(rhos[3, :, 0, 1:7], axis=1)
avg3std = np.std(rhos[3, :, 0, 1:7], axis=1)
rhoplot[3, 0].plot(avg3, color='black')
rhoplot[3, 0].errorbar(xs, avg3, avg3std)
#rhoplot[3, 0].errorbar(xs, meanoverruns, sem)
rhoplot[3, 0].set_title("Run 3 : beta 13", size=9)
rhoplot[3, 1].scatter(listbyruns[3, 0:7, :, 0], listbyruns[3, 0:7, :, 1], edgecolors="black", label='Run 3')
rhoplot[3, 1].legend()

avg4 = np.mean(rhos[4, :, 0, 1:7], axis=1)
avg4std = np.std(rhos[4, :, 0, 1:7], axis=1)
rhoplot[4, 0].plot(avg4, color='black')
rhoplot[4, 0].errorbar(xs, avg4, avg4std)
#rhoplot[3, 0].errorbar(xs, meanoverruns, sem)
rhoplot[4, 0].set_title("Run 4 : beta 14", size=9)
rhoplot[4, 1].scatter(listbyruns[4, 0:7, :, 0], listbyruns[4, 0:7, :, 1], edgecolors="black", label='Run 4')
rhoplot[4, 1].legend()


plt.legend()
plt.show()

plt.title("Rho adaptation on Runs 1 through 4")
plt.errorbar(xs, avg0, avg0std, label='run0beta: 10')
plt.errorbar(xs, avg1, avg1std, label='run1beta: 11')
plt.errorbar(xs, avg2, avg2std, label='run2beta: 12')
plt.errorbar(xs, avg3, avg3std, label='run3beta: 13')
plt.errorbar(xs, avg4, avg4std, label='run3beta: 14')
plt.xlabel("world states")
plt.ylabel("rhos")
plt.legend()
plt.show()

#plotting rhos. comparing different sigma values
#try generate individual errorbars for each plot
#taking all worldstates, single run, neuron 0 is paired with all others

rcParams['figure.figsize'] = 15, 15
fig, rhoplot = plt.subplots(4, 2)
plt.suptitle("Comparing RHO values for different Sigma Values[pxw_sigmas] Beta is 10 [Mus1]. N0 - N1:7", size=15)
avg5 = np.mean(rhos[5, :, 0, 1:7], axis=1)
avg5std = np.std(rhos[5, :, 0, 1:7], axis=1)
rhoplot[0, 0].plot(avg5, color='black')
rhoplot[0, 0].errorbar(xs, avg5, avg5std)
#rhoplot[0, 0].errorbar(xs, meanoverruns, sem)
rhoplot[0, 0].set_title("Run 05 : sigma = 0.02", size=9)
rhoplot[0, 1].scatter(listbyruns[5, 0:7, :, 0], listbyruns[5, 0:7, :, 1], edgecolors="black", label='Run 5')
rhoplot[0, 1].legend()

avg6 = np.mean(rhos[6, :, 0, 1:7], axis=1)
avg6std = np.std(rhos[6, :, 0, 1:7], axis=1)
rhoplot[1, 0].plot(avg6, color='black')
rhoplot[1, 0].errorbar(xs, avg6, avg6std)
#rhoplot[1, 0].errorbar(xs, meanoverruns, sem)
rhoplot[1, 0].set_title("Run 06 : sigma - 0.04", size=9)
rhoplot[1, 1].scatter(listbyruns[6, 0:7, :, 0], listbyruns[6, 0:7, :, 1], edgecolors="black", label='Run 6')
rhoplot[1, 1].legend()

avg7 = np.mean(rhos[7, :, 0, 1:7], axis=1)
avg7std = np.std(rhos[7, :, 0, 1:7], axis=1)
rhoplot[2, 0].plot(avg7, color='black')
rhoplot[2, 0].errorbar(xs, avg7, avg7std)
#rhoplot[2, 0].errorbar(xs, meanoverruns, sem)
rhoplot[2, 0].set_title("Run 07 : sigma - 0.07", size=9)
rhoplot[2, 1].scatter(listbyruns[7, 0:7, :, 0], listbyruns[7, 0:7, :, 1], edgecolors="black", label='Run 7')
rhoplot[2, 1].legend()

avg8 = np.mean(rhos[8, :, 0, 1:7], axis=1)
avg8std = np.std(rhos[8, :, 0, 1:7], axis=1)
rhoplot[3, 0].plot(avg8, color='black')
rhoplot[3, 0].errorbar(xs, avg8, avg8std)
#rhoplot[3, 0].errorbar(xs, meanoverruns, sem)
rhoplot[3, 0].set_title("Run 08 : sigma 0.11", size=9)
rhoplot[3, 1].scatter(listbyruns[8, 0:7, :, 0], listbyruns[8, 0:7, :, 1], edgecolors="black", label='Run 8')
rhoplot[3, 1].legend()

plt.legend()
plt.show()

plt.title("Rho adaptation on Runs 5 through 8 (beta is 10 and mus is default)")
plt.errorbar(xs, avg5, avg5std, label='run5sigma: 0.02')
plt.errorbar(xs, avg6, avg6std, label='run6sigma: 0.04')
plt.errorbar(xs, avg7, avg7std, label='run7sigma: 0.07')
plt.errorbar(xs, avg8, avg8std, label='run8sigma: 0.11')
plt.xlabel("world states")
plt.ylabel("rhos")
plt.legend()
plt.show()

rcParams['figure.figsize'] = 5, 5

plt.plot(xs, meanoverruns)
plt.errorbar(xs, meanoverruns, sem)
plt.title("Errorbar - All pairs of N0")
plt.show()

rcParams['figure.figsize'] = 15, 15
fig, mvplot = plt.subplots(3)
plt.suptitle("Multivariate Normal - WorldStates (Pair N0 and N1) (Run 12, 10 and Run 14)Tuning Curve mus3 vs mus1 vs mus5", size=14)
mvplot[0].scatter(listbyruns[12, 0, :, 0], listbyruns[12, 0, :, 1], edgecolors="black", color='red', label='W1')
mvplot[0].scatter(listbyruns[12, 1, :, 0], listbyruns[12, 1, :, 1], edgecolors="black", color='green', label="W2")
mvplot[0].scatter(listbyruns[12, 2, :, 0], listbyruns[12, 2, :, 1], edgecolors="black", color='violet', label="W3")
mvplot[0].scatter(listbyruns[12, 3, :, 0], listbyruns[12, 3, :, 1], edgecolors="black", color='indigo', label="W4")
mvplot[0].scatter(listbyruns[12, 4, :, 0], listbyruns[12, 4, :, 1], edgecolors="black", color='cyan', label="W5")
mvplot[0].scatter(listbyruns[12, 5, :, 0], listbyruns[12, 5, :, 1], edgecolors="black", color='brown', label="W6")
mvplot[0].scatter(listbyruns[12, 6, :, 0], listbyruns[12, 6, :, 1], edgecolors="black", color='orange', label="W7")
mvplot[0].scatter(listbyruns[12, 7, :, 0], listbyruns[12, 7, :, 1], edgecolors="black", color='yellow', label="W8")
mvplot[0].set_xlabel("Neuron 0")
mvplot[0].set_ylabel("Neuron 1")
mvplot[0].legend()

mvplot[1].scatter(listbyruns[10, 0, :, 0], listbyruns[10, 0, :, 1], edgecolors="black", color='red', label='W1')
mvplot[1].scatter(listbyruns[10, 1, :, 0], listbyruns[10, 1, :, 1], edgecolors="black", color='green', label="W2")
mvplot[1].scatter(listbyruns[10, 2, :, 0], listbyruns[10, 2, :, 1], edgecolors="black", color='violet', label="W3")
mvplot[1].scatter(listbyruns[10, 3, :, 0], listbyruns[10, 3, :, 1], edgecolors="black", color='indigo', label="W4")
mvplot[1].scatter(listbyruns[10, 4, :, 0], listbyruns[10, 4, :, 1], edgecolors="black", color='cyan', label="W5")
mvplot[1].scatter(listbyruns[10, 5, :, 0], listbyruns[10, 5, :, 1], edgecolors="black", color='brown', label="W6")
mvplot[1].scatter(listbyruns[10, 6, :, 0], listbyruns[10, 6, :, 1], edgecolors="black", color='orange', label="W7")
mvplot[1].scatter(listbyruns[10, 7, :, 0], listbyruns[10, 7, :, 1], edgecolors="black", color='yellow', label="W8")
mvplot[1].set_xlabel("Neuron 0")
mvplot[1].set_ylabel("Neuron 1")
mvplot[1].legend()

mvplot[2].scatter(listbyruns[14, 0, :, 0], listbyruns[14, 0, :, 1], edgecolors="black", color='red', label='W1')
mvplot[2].scatter(listbyruns[14, 1, :, 0], listbyruns[14, 1, :, 1], edgecolors="black", color='green', label="W2")
mvplot[2].scatter(listbyruns[14, 2, :, 0], listbyruns[14, 2, :, 1], edgecolors="black", color='violet', label="W3")
mvplot[2].scatter(listbyruns[14, 3, :, 0], listbyruns[14, 3, :, 1], edgecolors="black", color='indigo', label="W4")
mvplot[2].scatter(listbyruns[14, 4, :, 0], listbyruns[14, 4, :, 1], edgecolors="black", color='cyan', label="W5")
mvplot[2].scatter(listbyruns[14, 5, :, 0], listbyruns[14, 5, :, 1], edgecolors="black", color='brown', label="W6")
mvplot[2].scatter(listbyruns[14, 6, :, 0], listbyruns[14, 6, :, 1], edgecolors="black", color='orange', label="W7")
mvplot[2].scatter(listbyruns[14, 7, :, 0], listbyruns[14, 7, :, 1], edgecolors="black", color='yellow', label="W8")
mvplot[2].set_xlabel("Neuron 0")
mvplot[2].set_ylabel("Neuron 1")
mvplot[2].legend()
plt.show()

rcParams['figure.figsize'] = 15, 15
fig, mvplot = plt.subplots(3)
plt.suptitle("Multivariate Normal - WorldStates (Pair N0 and N1) (Run10 vs 12 vs Run 14)Tuning Curve mus1 vs mus3 vs mus5", size=14)
mvplot[0].scatter(listbyruns[10, 0, :, 0], listbyruns[10, 0, :, 5], edgecolors="black", color='red', label='W1')
mvplot[0].scatter(listbyruns[10, 1, :, 0], listbyruns[10, 1, :, 5], edgecolors="black", color='green', label="W2")
mvplot[0].scatter(listbyruns[10, 2, :, 0], listbyruns[10, 2, :, 5], edgecolors="black", color='violet', label="W3")
mvplot[0].scatter(listbyruns[10, 3, :, 0], listbyruns[10, 3, :, 5], edgecolors="black", color='indigo', label="W4")
mvplot[0].scatter(listbyruns[10, 4, :, 0], listbyruns[10, 4, :, 5], edgecolors="black", color='cyan', label="W5")
mvplot[0].scatter(listbyruns[10, 5, :, 0], listbyruns[10, 5, :, 5], edgecolors="black", color='brown', label="W6")
mvplot[0].scatter(listbyruns[10, 6, :, 0], listbyruns[10, 6, :, 5], edgecolors="black", color='orange', label="W7")
mvplot[0].scatter(listbyruns[10, 7, :, 0], listbyruns[10, 7, :, 5], edgecolors="black", color='yellow', label="W8")
mvplot[0].set_xlabel("Neuron 0")
mvplot[0].set_ylabel("Neuron 1")
mvplot[0].legend()

mvplot[1].scatter(listbyruns[12, 0, :, 0], listbyruns[12, 0, :, 6], edgecolors="black", color='red', label='W1')
mvplot[1].scatter(listbyruns[12, 1, :, 0], listbyruns[12, 1, :, 6], edgecolors="black", color='green', label="W2")
mvplot[1].scatter(listbyruns[12, 2, :, 0], listbyruns[12, 2, :, 6], edgecolors="black", color='violet', label="W3")
mvplot[1].scatter(listbyruns[12, 3, :, 0], listbyruns[12, 3, :, 6], edgecolors="black", color='indigo', label="W4")
mvplot[1].scatter(listbyruns[12, 4, :, 0], listbyruns[12, 4, :, 6], edgecolors="black", color='cyan', label="W5")
mvplot[1].scatter(listbyruns[12, 5, :, 0], listbyruns[12, 5, :, 6], edgecolors="black", color='brown', label="W6")
mvplot[1].scatter(listbyruns[12, 6, :, 0], listbyruns[12, 6, :, 6], edgecolors="black", color='orange', label="W7")
mvplot[1].scatter(listbyruns[12, 7, :, 0], listbyruns[12, 7, :, 6], edgecolors="black", color='yellow', label="W8")
mvplot[1].set_xlabel("Neuron 0")
mvplot[1].set_ylabel("Neuron 1")
mvplot[1].legend()

mvplot[2].scatter(listbyruns[14, 0, :, 0], listbyruns[14, 0, :, 7], edgecolors="black", color='red', label='W1')
mvplot[2].scatter(listbyruns[14, 1, :, 0], listbyruns[14, 1, :, 7], edgecolors="black", color='green', label="W2")
mvplot[2].scatter(listbyruns[14, 2, :, 0], listbyruns[14, 2, :, 7], edgecolors="black", color='violet', label="W3")
mvplot[2].scatter(listbyruns[14, 3, :, 0], listbyruns[14, 3, :, 7], edgecolors="black", color='indigo', label="W4")
mvplot[2].scatter(listbyruns[14, 4, :, 0], listbyruns[14, 4, :, 7], edgecolors="black", color='cyan', label="W5")
mvplot[2].scatter(listbyruns[14, 5, :, 0], listbyruns[14, 5, :, 7], edgecolors="black", color='brown', label="W6")
mvplot[2].scatter(listbyruns[14, 6, :, 0], listbyruns[14, 6, :, 7], edgecolors="black", color='orange', label="W7")
mvplot[2].scatter(listbyruns[14, 7, :, 0], listbyruns[14, 7, :, 7], edgecolors="black", color='yellow', label="W8")
mvplot[2].set_xlabel("Neuron 0")
mvplot[2].set_ylabel("Neuron 1")
mvplot[2].legend()
plt.show()


plt.title("Multivariate Normal - WorldStates (Pair N0 and N1) Run 10")
plt.scatter(listbyruns[10, 0, :, 0], listbyruns[10, 0, :, 1], edgecolors="black", color='red', label='W1')
plt.scatter(listbyruns[10, 1, :, 0], listbyruns[10, 1, :, 1], edgecolors="black", color='green', label="W2")
plt.scatter(listbyruns[10, 2, :, 0], listbyruns[10, 2, :, 1], edgecolors="black", color='violet', label="W3")
plt.scatter(listbyruns[10, 3, :, 0], listbyruns[10, 3, :, 1], edgecolors="black", color='indigo', label="W4")
plt.scatter(listbyruns[10, 4, :, 0], listbyruns[10, 4, :, 1], edgecolors="black", color='cyan', label="W5")
plt.scatter(listbyruns[10, 5, :, 0], listbyruns[10, 5, :, 1], edgecolors="black", color='brown', label="W6")
plt.scatter(listbyruns[10, 6, :, 0], listbyruns[10, 6, :, 1], edgecolors="black", color='orange', label="W7")
plt.scatter(listbyruns[10, 7, :, 0], listbyruns[10, 7, :, 1], edgecolors="black", color='yellow', label="W8")
plt.xlabel("Neuron 0")
plt.ylabel("Neuron 1")
plt.legend()
plt.show()

plt.scatter(listbyruns[10, 0:7, :, 0], listbyruns[10, 0:7, :, 1], edgecolors="black", label='Run 10[mus1]')
plt.scatter(listbyruns[11, 0:7, :, 0], listbyruns[11, 0:7, :, 1], edgecolors="black", label='Run 11[mus2]')
plt.scatter(listbyruns[12, 0:7, :, 0], listbyruns[12, 0:7, :, 1], edgecolors="black", label='Run 12[mus3]')
plt.scatter(listbyruns[13, 0:7, :, 0], listbyruns[13, 0:7, :, 1], edgecolors="black", label='Run 13[mus4]')
plt.scatter(listbyruns[14, 0:7, :, 0], listbyruns[14, 0:7, :, 1], edgecolors="black", label='Run 14[mus5]')
plt.xlabel("Neuron 0")
plt.ylabel("Neuron 1")
plt.legend()
plt.title("Multivariate Normal - Comparing Worldstates, Different Tuning Curves Beta 10 sigma 0.02")
plt.show()

plt.scatter(listbyruns[10, 0:7, :, 0], listbyruns[10, 0:7, :, 6], edgecolors="black", label='Run 10[mus1]')
plt.scatter(listbyruns[11, 0:7, :, 0], listbyruns[11, 0:7, :, 6], edgecolors="black", label='Run 11[mus2]')
plt.scatter(listbyruns[12, 0:7, :, 0], listbyruns[12, 0:7, :, 6], edgecolors="black", label='Run 12[mus3]')
plt.scatter(listbyruns[13, 0:7, :, 0], listbyruns[13, 0:7, :, 6], edgecolors="black", label='Run 13[mus4]')
plt.scatter(listbyruns[14, 0:7, :, 0], listbyruns[14, 0:7, :, 6], edgecolors="black", label='Run 14[mus5]')
plt.xlabel("Neuron 0")
plt.ylabel("Neuron 1")
plt.legend()
plt.title("Multivariate Normal - Comparing Worldstates, Different Tuning Curves Beta 10 sigma 0.02")
plt.show()

plt.scatter(listbyruns[0, 0:7, :, 0], listbyruns[0, 0:7, :, 1], edgecolors="black", label='Run 0 B-10')
plt.scatter(listbyruns[1, 0:7, :, 0], listbyruns[1, 0:7, :, 1], edgecolors="black", label='Run 1 B-11')
plt.scatter(listbyruns[2, 0:7, :, 0], listbyruns[2, 0:7, :, 1], edgecolors="black", label='Run 2 B-12')
plt.scatter(listbyruns[3, 0:7, :, 0], listbyruns[3, 0:7, :, 1], edgecolors="black", label='Run 3 B-13')
plt.scatter(listbyruns[4, 0:7, :, 0], listbyruns[4, 0:7, :, 1], edgecolors="black", label='Run 4 B-14')
plt.xlabel("Neuron 0")
plt.ylabel("Neuron 1")
plt.legend()
plt.title("Multivariate Normal - Comparing Worldstates, Different Betas Neuron 0 against Neuron 1")
plt.show()

plt.scatter(listbyruns[0, 0:7, :, 0], listbyruns[0, 0:7, :, 6], edgecolors="black", label='Run 0 B-10')
plt.scatter(listbyruns[1, 0:7, :, 0], listbyruns[1, 0:7, :, 6], edgecolors="black", label='Run 1 B-11')
plt.scatter(listbyruns[2, 0:7, :, 0], listbyruns[2, 0:7, :, 6], edgecolors="black", label='Run 2 B-12')
plt.scatter(listbyruns[3, 0:7, :, 0], listbyruns[3, 0:7, :, 6], edgecolors="black", label='Run 3 B-13')
plt.scatter(listbyruns[4, 0:7, :, 0], listbyruns[4, 0:7, :, 6], edgecolors="black", label='Run 4 B-14')
plt.xlabel("Neuron 0")
plt.ylabel("Neuron 1")
plt.legend()
plt.title("Multivariate Normal - Comparing Worldstates, Different Betas Neuron 4 against Neuron 6")
plt.show()


plt.scatter(listbyruns[5, 0:7, :, 0], listbyruns[5, 0:7, :, 1], edgecolors="black", label='Run 5 Sigma 0.02')
plt.scatter(listbyruns[6, 0:7, :, 0], listbyruns[6, 0:7, :, 1], edgecolors="black", label='Run 6 Sigma 0.04')
plt.scatter(listbyruns[7, 0:7, :, 0], listbyruns[7, 0:7, :, 1], edgecolors="black", label='Run 7 Sigma 0.07')
plt.scatter(listbyruns[8, 0:7, :, 0], listbyruns[8, 0:7, :, 1], edgecolors="black", label='Run 8 Sigma 0.11')
plt.scatter(listbyruns[9, 0:7, :, 0], listbyruns[9, 0:7, :, 1], edgecolors="black", label='Run 9 Sigma 0.16')
plt.xlabel("Neuron 0")
plt.ylabel("Neuron 1")
plt.legend()
plt.title("Multivariate Normal - Comparing Worldstates, Different Sigmas.[Beta - 10] Neuron 0 against Neuron 1")
plt.show()

plt.scatter(listbyruns[5, 0:7, :, 0], listbyruns[5, 0:7, :, 6], edgecolors="black", label='Run 5 Sigma 0.02')
plt.scatter(listbyruns[6, 0:7, :, 0], listbyruns[6, 0:7, :, 6], edgecolors="black", label='Run 6 Sigma 0.04')
plt.scatter(listbyruns[7, 0:7, :, 0], listbyruns[7, 0:7, :, 6], edgecolors="black", label='Run 7 Sigma 0.07')
plt.scatter(listbyruns[8, 0:7, :, 0], listbyruns[8, 0:7, :, 6], edgecolors="black", label='Run 8 Sigma 0.11')
plt.scatter(listbyruns[9, 0:7, :, 0], listbyruns[9, 0:7, :, 6], edgecolors="black", label='Run 9 Sigma 0.16')
plt.xlabel("Neuron 0")
plt.ylabel("Neuron 1")
plt.legend()
plt.title("Multivariate Normal - Comparing Worldstates, Different Sigmas.[Beta - 10] Neuron 4 against Neuron 6")
plt.show()
sys.exit(0)
import numpy as np
import scipy as sp
import scipy.special as sps
import matplotlib.pyplot as plt
import os
import math

"""utility"""
def gaussn(x):
	return np.exp(-x**2/2)/np.sqrt(2*np.pi)
def gauss(x,mu,sigma):
	return gaussn((x-mu)/float(sigma))/float(sigma)
def gauss_utility(N,K,sigma):
	util = np.zeros((N,K))
	for n in range(0,N):
		for k in range(0,K):
			x = np.arange(0,K,1)
			util[n,:] = gauss(np.arange(0,K),n,sigma)
	u =  util/np.max(util)
	return u


"""dkl"""
def calc_dkl(p,q):
	dkl =0
	for w in range(len(p)):
		dkl += kl_div(p[w,:],q)
	return dkl
def calc_dkl_w(p,q):
	dkl_w = []
	for w in range(len(p)):
		dkl_w.append(kl_div(p[w,:],q))
	return np.array(dkl_w)
def kl_div(p,q):
	eps = 1e-16
	p+=eps
	q+=eps
	dkl = np.einsum('i,i->',p, np.log2(p/q))
	#dkl = np.einsum('i,i->',p,(np.log(p/q)/np.log(2)))
	return dkl



# """plots"""
def plot_bivariatenormal(state_size, i, D, pxw_means):
    fig = plt.figure(figsize=(10, 10))
    if len(D):
	    for w in range(state_size):
	        plt.scatter(D[w,:,i[0]],D[w,:,i[1]],alpha=0.5,label=w)
    plt.scatter(pxw_means[:,i[0]],pxw_means[:,i[1]],c='k')
    plt.xlabel('$x_'+str(i[0])+'$', fontsize=20)
    plt.ylabel('$x_'+str(i[1])+'$', fontsize=20)
    plt.legend(title='w',numpoints=1)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    return fig


def plot_means_w(state_size, i, pxw_means):
    fig = plt.figure(0)
    for x in i:
        plt.scatter(range(state_size),pxw_means[:,x],label='\\mu_{x_'+str(x)+'}')
    plt.xlabel('w')
    plt.ylabel('$\\mu$')
    return fig


# Population Vector ( p(a|x)- model )
def pol2cart(rho, phi):
    x = rho * math.cos(math.radians(phi))
    y = rho * math.sin(math.radians(phi))
    return(x, y)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return phi

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_phis(action_size,x_means):
    a_phis = np.array([i*(360/(action_size)) for i in range(action_size)])# phis for all action directories
    phis = np.zeros((np.shape(x_means))) # phis for all x
    for a in range(action_size):# negative angles
        if a_phis[a]>180:
            a_phis[a] = -(360-a_phis[a])
    for i in range(len(phis)):
        phis[i] = a_phis[int(x_means[i])]
    return phis

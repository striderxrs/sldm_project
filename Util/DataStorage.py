import numpy as np
import os
import pickle

class DataStorage(object):

    def __init__(self, name="",trials=1, episodes=1, state_size=8, action_size=8, betas=[100,100], lrs=[0.1,0.1], steps=[1,1], x_means=[1.0,1.0], x_sigmas =[1.0,1.0]):
        self.name = name
        self.episodes = episodes
        self.trials = trials
        self.betas = betas
        self.lrs = lrs
        self.steps = steps    # update steps GD/NN
        self.state_size = state_size
        self.action_size = action_size
        self.x_means = x_means
        self.x_sigmas = x_sigmas
        self.pxw_means = np.zeros((state_size,len(x_means)))
        self.pxw_sigmas = np.zeros((state_size,len(x_means)))
        self.reset_episodes()

    def reset_episodes(self):
        self.dkl_w_e = []
        self.dkl_e = []
        self.I_X_A_e = []
        self.obj_grad = []
        self.rhos_e = []
        self.exp_u_e = []
        self.pxw_sigmas_e = []
        self.pxw_means_e = []
        self.p_a_w_e = []
        self.p_a_e = []

    def update(self, attribute, data, index=None, scale=False):
        arr = getattr(self, attribute)
        if type(arr) is list:
            arr.append(np.array(data)) #not as a list but an array
        else:
            if len(arr.shape) == 3:
                arr[index, :, :] = data
            else:
                arr[index, :] = data
            if scale:
                if len(arr.shape) == 3:
                    min = np.min(arr[index, :, :])
                    max = np.max(arr[index, :, :])
                    arr[index, :, :] -= min
                    arr[index, :, :] /= (max - min)
                else:
                    min = np.min(arr[index, :])
                    max = np.max(arr[index, :])
                    arr[index, :] -= min
                    arr[index, :] /= (max - min)


    def save(self):
        file = open(self.name, mode='wb')
        file.write(pickle.dumps(self,protocol=pickle.HIGHEST_PROTOCOL))
        file.close()
    def save_model(self,model):
        model[0].save(self.name+'_action_model0.h5')
        model[1].save(self.name+'_action_model1.h5')

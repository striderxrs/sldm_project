"""
Serial Gradient Agent
Noise correlation for neuron population with N=8 (multivariate normal distribution)
update parameters p(x|w): rho
"""

from Agent import *
from Util.helper import *
from Util.DataStorage import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import logging
logger = tf.get_logger()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger.setLevel(logging.ERROR)
#tf.logging.set_verbosity(tf.logging.ERROR)

"""########## SETUP ##########"""
"""environment"""
state_size = 8
action_size = state_size
utility = gauss_utility(state_size,action_size,0.000001) #sigma is 3rd, reward for monkey
max_trials = 500


"""Neuron Population"""
x_size = 8  # number neurons
x_means = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]    # neurons preferred directions
n=1         # mult.factor
x_size =x_size*n
x_means = np.tile(x_means,n)


"""########## FUNCTIONS ##########"""
def solve(nb_episodes, beta_1, beta_2, lr1, lr2, x_means, sigma, w_episodes):
    print("betas:", beta_1, beta_2)
    with tf.Session(graph=tf.get_default_graph()) as sess:
        timestamp = time.strftime("%m_%d-%H_%M_%S")
        b = str(int(beta_1))
        b = b.replace(".", "")
        s = str(sigma)
        s = s.replace(".","")
        foldername =  os.path.join('results','test')
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        name = os.path.join(os.getcwd(),foldername, timestamp)
        storage = DataStorage(name=name, episodes=nb_episodes, state_size=state_size, action_size=action_size, betas=[beta_1,beta_2], lrs=[lr1,lr2], x_means=x_means)

        agent = Agent(state_size=state_size,
                        x_size = x_size,
                        action_size=action_size,
                        utility=utility,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        lr1=lr1,
                        lr2=lr2,
                        x_means = x_means,
                        sigma= sigma,
                        sess=sess,
                        storage=storage,
                        w_episodes=w_episodes)

        for e in range(nb_episodes):
            # """generate training data for action_net (-> memory)"""
            # for w in range(state_size):
            #     mem = agent.make_prediction(w, max_trials)
            agent.fit_model()
            agent.reset_memory()
            agent.episode_over()

            """evaluate"""
            _, _, exp_u, dkl = agent.get_posterior_ps()

            print("E[U]", exp_u)
            print("DKL", dkl)
    return agent, storage


def run(nb_episodes, beta_1, beta_2, lr1, lr2, x_means, sigma, w_episodes):
    storages=[]
    for i in range(len(beta_1)): # solve for different sigmas
        agent,storage = solve(nb_episodes=nb_episodes,
                         beta_1=beta_1[i],
                         beta_2=beta_2[i],
                         lr1=lr1,
                         lr2=lr2,
                         x_means=x_means,
                         sigma=sigma,
                         w_episodes=w_episodes)
        storage.save()  #put it in solve to update after every run, episode storage
        storages.append(storage)
    return agent, storages

"""########## RUN ##########"""
nb_episodes = 100
beta_1 = [60.0]
beta_2 = [200.0]
lr1 = 5e-3
lr2 = 1e-3                  # not used
w_episodes = 1
sigma = 0.02

agent,storages = run(nb_episodes=nb_episodes, beta_1=beta_1, beta_2=beta_2, lr1=lr1, lr2=lr2, x_means=x_means,sigma=sigma, w_episodes=w_episodes)


# def run(nb_episodes, beta_1, beta_2, lr1, lr2, x_means, sigma, w_episodes):
#     storages=[]
#     for i in range(len(beta_1)): # solve for different betas
#         agent,storage = solve(nb_episodes=nb_episodes,
#                          beta_1=beta_1[i],
#                          beta_2=beta_2[i],
#                          lr1=lr1,
#                          lr2=lr2,
#                          x_means=x_means,
#                          sigma=sigma,
#                          w_episodes=w_episodes)
#         storage.save()  #put it in solve to update after every run, episode storage
#         storages.append(storage)
#     return agent, storages
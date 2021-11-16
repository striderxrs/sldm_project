from Util.helper import *
from scipy.stats import multivariate_normal
import emcee
import tensorflow as tf
from keras.models import Model, load_model
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import os

eps = 1e-16

class Agent():
    def __init__(self,
                 state_size,
                 x_size,
                 action_size,
                 utility=None,
                 beta_1=1,
                 beta_2=1,
                 lr1=0.01,
                 lr2=0.01,
                 x_means=[0.0,0.0],
                 sigma=0.1,
                 sess=None,
                 storage=None,
                 w_episodes=1):
        """
        This class represent an bounded rational agent that implements the "Serial Information Processing
        Hierarchies" according to

        "Bounded Rationality, Abstraction, and Hierarchical Decision-Making: An Information-Theoretic
        Optimality Principle" by Genewein et al."

        applied to
        "Selective Changes in Noise Correlations Contribute to an Enhanced Representation of Saccadic Targets
        in Prefrontal Neuronal Ensembles by Dehaqani et al."

        :param state_size: number of gaze directions
        :param x_size: dimension of x (number of neuron-population)
        :param action_size: number of gaze actions
        :param beta_1 and beta_2: \beta parameters for optimization (see paper)
        :param lr1 and lr2: learning rates
        :param x_means: neurons preferred directions
        :param sigma: pxw_sigma (for all w same)
        :param sess: tf.Session instance
        :param storage: DataStorage instance
        """
        self.state_size = state_size
        self.x_size = x_size            # neuron population
        self.action_size = action_size
        self.x_means = x_means          # preferred directions
        self.memory = []                # for NN update
        self.utility = utility
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lr1 = lr1
        self.lr2 = lr2
        self.pxw_means = np.zeros((self.state_size,self.x_size))
        self.pxw_sigmas = np.full((self.state_size,self.x_size),sigma)

        self.rho = np.zeros((self.state_size,self.x_size,self.x_size))

        self.p_w = np.ones(shape=(self.state_size)) / self.state_size
        self.p_a_w = np.ones(shape=(self.state_size, self.action_size))/(self.action_size*self.state_size)
        self.p_a = np.ones(shape=(self.action_size))/self.action_size
        self.x_model = None
        self.action_model = None

        self.episodes = 0
        self.w_episodes = w_episodes

        """tensorflow session and tensors"""
        self.sess = sess
        self.writer = None
        self.w_tf = None
        self.F = None
        self.D = None
        self.px = None
        self.theta_tf = None
        self.eta_tf = None
        self.obj_grad = None

        """storage variables"""
        self.storage = storage
        self.dkl_w = []
        self.dkl = 0
        self.I_X_A_w = np.zeros((self.state_size))
        self.exp_u = 0
        self.obj = 0
        self.build_models()            # initialize models

    def __del__(self):
        pass


#####################################
    def build_models(self):
        """x model"""
        self.x_model = self.build_x_model()                 # model to learn p(x|w)
        self.init_sgd()
        """"action model"""
        self.action_model = self.build_action_model()       # model to learn p(a|x)

    def covariance(self,w):
        cov = np.einsum('i,j,ij->ij',self.pxw_sigmas[w],self.pxw_sigmas[w],self.rho[w]+np.eye(self.x_size))
        return cov

    def covariance_tf(self):
        w = self.w_tf
        cov = tf.einsum('i,j,ij->ij',self.pxw_sigmas_tf[w],self.pxw_sigmas_tf[w],self.rho_w_tf+np.eye(self.x_size))
        return cov

    def set_mus(self):
        #mus = [0.80, 0.57, 0.47, 0.43, 0.40, 0.43, 0.47, 0.57]           # example tuning curve shape (maximum shifted to preferred direction of each neuron)
        #mus = [0.10, 0.12, 0.20, 0.40, 0.20, 0.12, 0.10, 0.05]
        #mus = [0.90, 0.40, 0.50, 0.68, 0.78, 0.68, 0.50, 0.40]
        #mus = [0.98, 0.80, 0.75, 0.70, 0.65, 0.70, 0.75, 0.80]
        mus = [0.95, 0.45, 0.38, 0.30, 0.22, 0.30, 0.38, 0.45]

        mus = mus[0:self.state_size]
        for i in range(self.x_size):
            self.pxw_means[:,i] = np.roll(mus, int(self.x_means[i]))

    def set_x_model(self,w):
        cov = self.covariance(w)
        def p_x_w_model(x):
            return multivariate_normal.pdf(x,mean=self.pxw_means[w],cov=cov)
        return p_x_w_model

    def build_x_model(self):
        self.set_mus()
        p_x_w_model = []
        for w in range(self.state_size):
            p_x_w_model.append(self.set_x_model(w))
        return p_x_w_model

    def population_vector(self, x_sample):
        F = sum(x_sample,0)
        F= F/np.linalg.norm(F)            # population vector
        return F

    def build_action_model(self):
        def pax_model(x):
            phis = get_phis(self.state_size,self.x_means)
            xy = np.array([pol2cart(x[i],phis[i]) for i in range(len(x))])  # xy coordinates
            "population_vector - categorical p(a|x) model"
            xy_PV = self.population_vector(xy)
            phi_PV = cart2pol(xy_PV[0],xy_PV[1])
            Fphi = np.rad2deg(phi_PV)
            A = find_nearest(phis, Fphi)
            "action probability (categorical)"
            p_a_x = np.zeros((self.state_size))
            p_a_x[A] = 1    # delta probability for action A
            return p_a_x
        return pax_model

    def init_sgd(self):
        self.theta_tf = tf.Variable(initial_value=np.arctanh(self.rho), dtype=tf.float32, name="rho_tfvar") #theta optimal for the whole range
        self.pxw_sigmas_tf = tf.constant(self.pxw_sigmas,dtype=tf.float32)
        self.pxw_means_tf = tf.constant(self.pxw_means,dtype=tf.float32)
        self.w_tf = tf.placeholder(name="w_tf",shape=(), dtype=tf.int32)
        self.D = tf.placeholder(name="D",shape=(None,self.x_size), dtype=tf.float32)
        self.F = tf.placeholder(name="F", shape=(None), dtype=tf.float32)
        self.px = tf.placeholder(name="px", shape=(None), dtype=tf.float32)
        self.rho_w_tf = self.extract_rho()

        obj_x_w_grad, obj_x_w = self.calc_obj_x_w()
        self.obj_grad = tf.reduce_mean(obj_x_w_grad)                # expected value over p_x_w approximated by x-samples
        self.obj = tf.reduce_mean(obj_x_w)

        opt = tf.train.AdamOptimizer(learning_rate=self.lr1, name='Adam')

        self.train_op = opt.minimize(-self.obj_grad)

        self.writer = SummaryWriter()

        init = tf.global_variables_initializer()
        self.sess.run(init)

#####################################
    def fit_model(self):
        print("pxw-model update: step ", (self.episodes+1))
        self.update_pxw()


#####################################
    def update_pxw(self, Nsamples=1000):
        obj_current = []
        obj_grad_current =[]
        D = []
        p_a_xs = []
        F = []
        for w in range(self.state_size):
            D = self.sample_from_p_x_w(w,Nsamples)
            p_a_xs = [self.predict_p_a_x(x) for x in D]
            F = [self.compute_F(pax,w) for pax in p_a_xs]
            px = self.calc_px(D)

            obj_current_w = []
            obj_grad_current_w = []
            for N in range(self.w_episodes):
                self.sess.run(self.train_op, feed_dict={self.D: D, self.F: F, self.w_tf: w, self.px: px}) # update theta_tf and rho_tf
                obj_current_w.append(self.obj.eval(session=self.sess, feed_dict={self.D: D, self.F: F, self.w_tf: w, self.px: px}))
                obj_grad_current_w.append(self.obj_grad.eval(session=self.sess, feed_dict={self.D: D, self.F: F, self.w_tf: w, self.px: px}))
                self.rho[w]= self.rho_w_tf.eval(session=self.sess, feed_dict={self.w_tf:w})              # update rho

            obj_current.append(np.mean(obj_current_w))
            obj_grad_current.append(np.mean(obj_grad_current_w))


        self.writer.add_scalar("obj_grad_mean", np.mean(obj_grad_current), global_step = self.episodes)
        self.writer.add_scalar("obj_mean", np.mean(obj_current), global_step = self.episodes)
        #self.writer.add_scalar("rhoo?", np.ptp(self.rho), global_step=self.x_size) #histogram?? tf.histogram?
        self.writer.add_scalar("rho_mean", np.mean(self.rho), global_step = self.episodes)
        self.x_model = self.build_x_model()                             # update p_x_w

    def compute_F(self,pax,w):
        I_X_A = kl_div(pax,self.p_a)
        EU = pax.dot(self.utility[w,:])
        F = EU # - (1.0/self.beta_2)*I_X_A   # infinitly high beta_2
        return F

    def calc_px(self,D):
        px = [np.mean([self.x_model[i](x) for i in range(self.state_size)]) for x in D]
        return px


    def extract_rho(self):
        ## extract symmetric rho from theta with 1s on diagonal
        theta_upperwdiag = tf.matrix_band_part(self.theta_tf[self.w_tf], 0, -1)
        theta_upperwdiag = theta_upperwdiag - tf.sign(theta_upperwdiag)*1e-4*tf.random.uniform(shape=tf.shape(theta_upperwdiag), minval=0,maxval=1)     # avoid matrix inverse error
        theta_diag = tf.matrix_band_part(self.theta_tf[self.w_tf], 0, 0)
        theta_upper = theta_upperwdiag - theta_diag
        rho = tf.tanh(theta_upper + tf.transpose(theta_upper))+tf.eye(self.x_size)
        return rho

    def calc_obj_x_w(self):
        cov = self.covariance_tf()
        D=self.D
        F=self.F

        means=self.pxw_means_tf[self.w_tf]
        log_p = -0.5*tf.math.reduce_sum(tf.multiply(D-means,tf.transpose(tf.matmul(tf.linalg.inv(cov),tf.transpose(D-means)))),axis=1)
        norm = tf.sqrt(tf.pow(2*np.pi,self.x_size)*tf.linalg.det(cov))

        log_p = log_p - (tf.log(norm)/np.log(2))
        obj_x_w_grad = log_p*(F - (1/self.beta_1)/2 * log_p - 1/self.beta_1- 1/self.beta_1 * tf.log(self.px))
        obj_x_w = F - (1/self.beta_1)*(log_p-tf.log(self.px))
        return obj_x_w_grad, obj_x_w

#####################################
    def compute_V_a_x(self,x):
        """
        utilities of actions for given x
        """
        p_x_wi = np.zeros((self.state_size))
        for w in range(0,self.state_size):
            p_x_wi[w] = self.x_model[w](x)
        z = np.sum(self.p_w * p_x_wi)
        p_w_x = p_x_wi*self.p_w / z
        V_a_x = np.einsum('i,ij->j',p_w_x,self.utility)
        return V_a_x

    def sample_from_p_x_w(self, w, Nsamples):
        """
        make x prediction for given w, sample from bivariate distribution (pxw_rho)
        """
        n = 0       # sample counter
        x_samples = []
        cov = self.covariance(w)
        means_w = self.pxw_means[w]
        while n<Nsamples:
            x_sample = multivariate_normal.rvs(mean=means_w, cov=cov, size=1)
            if x_sample[0]>0 and x_sample[1]>0:
                x_samples.append(x_sample)
                n+=1
        return np.array(x_samples)

    def predict_p_a_x(self, x):
        """
        make a prediction for given x
        """
        p_a_x = self.action_model(x)
        return p_a_x


    def update_probabilities(self, Nsamples=1000):
        """
        Update p_a_w, p_a and p_x?
        """
        p_x_w = np.zeros((self.state_size, Nsamples))
        p_a_x = np.zeros((self.state_size, Nsamples, self.action_size))
        I_X_A = np.zeros((self.state_size))
        for w in range(self.state_size):
            x_samples = self.sample_from_p_x_w(w, Nsamples)
            x_samples = np.maximum(0.0, x_samples)
            p_a_xs = [self.predict_p_a_x(x) for x in x_samples]
            p_a_x[w] = p_a_xs
            self.p_a_w[w] = np.array(np.mean(p_a_xs,0))     # average over x_samples
        alpha = 0
        n = self.episodes+1
        t = 1-alpha**n
        self.p_a = t*np.einsum('i, ij->j', self.p_w, self.p_a_w) + (1-t)*self.p_a       # alte samples ueber Zeit immer weniger beruecksichtigt
        for w in range(self.state_size):
            dkls = [kl_div(pax, self.p_a) for pax in p_a_x[w]]
            self.I_X_A_w[w] = np.mean(dkls)                 # mean over samples

    def get_posterior_ps(self):
        self.update_probabilities()
        exp_u = np.einsum('i,ij,ij->',self.p_w,self.utility,self.p_a_w)
        self.exp_u = exp_u
        self.dkl = calc_dkl(self.p_a_w, self.p_a)
        self.dkl_w = calc_dkl_w(self.p_a_w, self.p_a)
        self.writer.add_scalar("EU", self.exp_u, self.episodes)
        self.writer.add_scalar("DKL", self.dkl, self.episodes)
        return self.p_a_w, self.p_a, self.exp_u, self.dkl

    def save_stats(self):
        self.storage.update(attribute='rhos_e',data=self.rho)
        self.storage.update(attribute='dkl_w_e',data=self.dkl_w)
        self.storage.update(attribute='dkl_e',data=self.dkl)
        self.storage.update(attribute='exp_u_e',data=self.exp_u)
        self.storage.update(attribute='p_a_w_e',data=self.p_a_w)
        self.storage.update(attribute='p_a_e',data=self.p_a)
        self.storage.update(attribute='I_X_A_e', data=list(self.I_X_A_w))
        self.storage.update(attribute='pxw_sigmas_e', data=self.pxw_sigmas)
        self.storage.update(attribute='pxw_means_e', data=self.pxw_means)
        return

    def episode_over(self):
        _,_,_,_ =self.get_posterior_ps()
        self.save_stats()
        self.episodes += 1

    def reset_memory(self):
        self.memory = []

    def remember_selection(self, x, p_a_x, objective,dkl,V_a_x):
        self.memory.append([x, p_a_x, objective,dkl,V_a_x])

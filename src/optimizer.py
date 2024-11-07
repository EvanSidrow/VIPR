import autograd.numpy as np
from autograd import grad
from autograd.scipy.stats import norm
#from autograd_gamma import gamma, gammainc, gammaincc, gammaincln, gammainccln

import matplotlib.pyplot as plt

from misc_funcs import logdotexp
from tree import Tree
from copy import deepcopy

class Optimizer:

    def __init__(self,tree_log_probs,theta):

        # optimization parameters
        self.alpha = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 10e-8

        # optimization values
        self.t = 0
        self.m = 0
        self.v = 0

        # datapop_size
        self.tree_log_probs = tree_log_probs

        # initial parameters for q
        self.theta = theta

        # values from optimization
        self.ELBO_ests = []

    def optimize_q_reparam(self,batch_size,iters,pop_size=1.0,alphas=None):

        self.t = 0
        n_species = self.tree_log_probs.shape[0]

        if alphas is not None:
            alphas = np.exp(np.linspace(np.log(alphas[0]),np.log(alphas[1]),iters))
        else:
            alphas = self.alpha*np.ones(iters)

        for i in range(iters):

            self.alpha = alphas[i]
            Zs = np.random.normal(size=(batch_size,n_species,n_species))

            # define function to estimate ELBO
            def ELBO_sample(theta):
                ELBO = 0
                for Z in Zs:
                    log_times = np.exp(theta[1])*Z+theta[0]
                    log_times = log_times + np.triu(np.full(n_species, np.inf))
                    tree = Tree(theta,log_times,deepcopy(self.tree_log_probs),
                                pop_size=pop_size)
                    ELBO_hat = (tree.log_p - tree.log_q)/batch_size
                    if not np.isnan(ELBO_hat):
                        ELBO = ELBO + ELBO_hat
                return ELBO

            # get estimate of ELBO and gradient
            ELBO_est = ELBO_sample(self.theta)
            grad_theta = grad(ELBO_sample)(self.theta)

            self.update_theta(grad_theta)

            self.ELBO_ests.append(ELBO_est)

            if iters >= 10 and i % int(iters/10) == 0:
                print(i/iters)
        return

    def update_theta(self,grad):

        self.t = self.t + 1

        # Adam optimizer
        self.m = self.beta1*self.m + (1.0-self.beta1)*grad
        m_hat = self.m/(1.0-self.beta1**self.t)

        self.v = self.beta2*self.v + (1.0-self.beta2)*np.square(grad)
        v_hat = self.v/(1.0-self.beta2**self.t)

        delta = self.alpha*m_hat/(np.sqrt(v_hat) + self.epsilon)

        # SGD optimizer
        #delta = grad*self.alpha/np.sqrt(self.t)
        #print("delta:",delta)

        # update
        self.theta = self.theta + delta

        return

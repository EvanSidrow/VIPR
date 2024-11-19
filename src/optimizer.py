import autograd.numpy as np
from autograd import grad
from autograd.scipy.stats import norm
#from autograd_gamma import gamma, gammainc, gammaincc, gammaincln, gammainccln
from autograd.scipy.special import logsumexp

import matplotlib.pyplot as plt
import datetime

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
        self.grad_norms = []

    def optimize_q_reparam(self,batch_size,iters,print_out=False,pop_size=1.0,alphas=None):

        start = datetime.datetime.now()
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
            self.grad_norms.append(np.sum(grad_theta**2))

            if iters >= 10 and i % int(iters/10) == 0:
                print(i/iters)
                print("runtime: ",datetime.datetime.now()-start)
                print("grad_norm: ",np.sum(grad_theta**2))
                if print_out:
                    print(grad_theta)
                    print(self.theta)
                print("")
        return

    def optimize_q_reinforce(self,batch_size,iters,print_out=False,pop_size=1.0,alphas=None):

        start = datetime.datetime.now()
        self.t = 0
        n_species = self.tree_log_probs.shape[0]

        if alphas is not None:
            alphas = np.exp(np.linspace(np.log(alphas[0]),np.log(alphas[1]),iters))
        else:
            alphas = self.alpha*np.ones(iters)

        for i in range(iters):

            self.alpha = alphas[i]

            # create reinforce estimator
            grad_q_hats = np.zeros((batch_size,2,n_species,n_species))
            ELBO_hats = np.zeros(batch_size)
            ELBO_est_total = 0

            # run through batches
            for j in range(batch_size):

                # sample tree
                Z = np.random.normal(size=(n_species,n_species))
                log_times = np.exp(self.theta[1])*Z+self.theta[0]
                log_times = log_times + np.triu(np.full(n_species, np.inf))

                # estimate log_q
                def log_q_sample(theta):
                    tree = Tree(theta,log_times,
                                deepcopy(self.tree_log_probs),
                                pop_size=pop_size)
                    return tree.log_q
                grad_q_hats[j] = grad(log_q_sample)(self.theta)

                # estimate ELBO
                tree = Tree(self.theta,log_times,
                            deepcopy(self.tree_log_probs),
                            pop_size=pop_size)
                ELBO_hats[j] = tree.log_p - tree.log_q

            weights = ELBO_hats - np.sum(ELBO_hats)/batch_size
            grad_theta = np.sum(weights[:,None,None,None]*grad_q_hats,axis=0)/(batch_size-1)
            self.update_theta(grad_theta)

            # update trace of run
            self.ELBO_ests.append(np.mean(ELBO_hats))
            self.grad_norms.append(np.sum(grad_theta**2))

            if iters >= 10 and i % int(iters/10) == 0:
                print(i/iters)
                print("runtime: ",datetime.datetime.now()-start)
                print("grad_norm: ",np.sum(grad_theta**2))
                if print_out:
                    print("grad_theta: ",grad_theta)
                    print("theta: ",self.theta)
                print("")
        return

    def optimize_q_VIMCO(self,batch_size,iters,print_out=False,pop_size=1.0,alphas=None):

        start = datetime.datetime.now()
        self.t = 0
        n_species = self.tree_log_probs.shape[0]

        if alphas is not None:
            alphas = np.exp(np.linspace(np.log(alphas[0]),np.log(alphas[1]),iters))
        else:
            alphas = self.alpha*np.ones(iters)

        for i in range(iters):

            self.alpha = alphas[i]

            # array of gradients
            grad_q_hats = np.zeros((batch_size,2,n_species,n_species))

            # create array of ELBO estimates
            ELBO_hats = np.zeros(batch_size)

            # run through batches
            for j in range(batch_size):

                # sample tree
                Z = np.random.normal(size=(n_species,n_species))
                log_times = np.exp(self.theta[1])*Z+self.theta[0]
                log_times = log_times + np.triu(np.full(n_species, np.inf))

                # estimate log_q
                def log_q_sample(theta):
                    tree = Tree(theta,log_times,
                                deepcopy(self.tree_log_probs),
                                pop_size=pop_size)
                    return tree.log_q
                grad_q_hats[j] = grad(log_q_sample)(self.theta)

                # estimate ELBO
                tree = Tree(self.theta,log_times,
                            deepcopy(self.tree_log_probs),
                            pop_size=pop_size)
                ELBO_hats[j] = tree.log_p - tree.log_q

            # calculate VIMCO estimator
            EBLO_logsumexp = logsumexp(ELBO_hats)
            L_hat_K = EBLO_logsumexp - np.log(n_species)
            w_tilde = np.exp(ELBO_hats - EBLO_logsumexp)

            ELBO_logsumexp_minus_j = np.zeros(batch_size)
            ELBO_hats_minus_j = (np.sum(ELBO_hats) - ELBO_hats)/(n_species-1)
            for j in range(batch_size):
                ELBO_logsumexp_minus_j[j] = logsumexp([x if i != j else ELBO_hats_minus_j[j] for i,x in enumerate(ELBO_hats)]) - np.log(n_species)

            L_hat_K_minus_j = L_hat_K - ELBO_logsumexp_minus_j

            grad_theta = np.sum((L_hat_K_minus_j - w_tilde)[:,None,None,None]*grad_q_hats,axis = 0)

            self.update_theta(grad_theta)

            # update trace of run
            self.ELBO_ests.append(np.mean(ELBO_hats))
            self.grad_norms.append(np.sum(grad_theta**2))

            if iters >= 10 and i % int(iters/10) == 0:
                print(i/iters)
                print("runtime: ",datetime.datetime.now()-start)
                print("grad_norm: ",np.sum(grad_theta**2))
                if print_out:
                    print(grad_theta)
                    print(self.theta)
                print("")
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

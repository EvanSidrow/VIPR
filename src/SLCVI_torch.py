import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

import time
import math
import random
import numpy as np
import datetime

from tree_torch import Tree


class SLCVI(nn.Module):

    def __init__(self,tree_log_probs,theta,pop_size):
        super().__init__()

        # set parameters
        self.theta = nn.Parameter(torch.tensor(theta, requires_grad=True))

        # set values for tree
        self.tree_log_probs = tree_log_probs
        self.n_species = tree_log_probs.shape[0]
        self.pop_size = pop_size

        # values from optimization
        self.optimize = None
        self.ELBO_ests = []
        self.grad_norms = []
        self.run_time = None
        self.run_times = []
        self.epochs = []

        #torch.set_num_threads(1)

    def ELBO_reparam(self,batch_size):

        ELBO = 0
        for _ in range(batch_size):
            Z = torch.normal(mean=0.0,std=1.0,size=(self.n_species,self.n_species))
            log_times = torch.exp(self.theta[1])*Z+self.theta[0]
            log_times = log_times + torch.triu(torch.full((self.n_species,self.n_species), float("Inf")))
            tree = Tree(self.theta,
                        log_times,
                        self.tree_log_probs.detach(),
                        pop_size=self.pop_size)
            ELBO_hat = (tree.log_p - tree.log_q)/batch_size
            ELBO = ELBO + ELBO_hat
        return ELBO

    def ELBO_reinforce(self,batch_size):

        # create reinforce estimator
        ELBO_hats = torch.zeros(batch_size)
        log_qs = torch.zeros(batch_size)

        # run through batches
        for i in range(batch_size):

            # sample tree
            Z = torch.normal(mean=0.0,std=1.0,size=(self.n_species,self.n_species))
            log_times = torch.exp(self.theta[1])*Z+self.theta[0]
            log_times = log_times + torch.triu(torch.full((self.n_species,self.n_species), float("Inf")))
            log_times = log_times.detach()
            tree = Tree(self.theta,
                        log_times,
                        self.tree_log_probs.detach(),
                        pop_size=self.pop_size)
            ELBO_hats[i] = (tree.log_p - tree.log_q).detach()
            log_qs[i] = tree.log_q

        weights = ELBO_hats - torch.mean(ELBO_hats)

        return torch.sum(weights*log_qs) / (batch_size-1)

    def ELBO_VIMCO(self,batch_size):

        # create VIMCO estimator
        ELBO_hats = torch.zeros(batch_size)
        log_qs = torch.zeros(batch_size)

        # run through batches
        for i in range(batch_size):

            # sample tree
            Z = torch.normal(mean=0.0,std=1.0,size=(self.n_species,self.n_species))
            log_times = torch.exp(self.theta[1])*Z+self.theta[0]
            log_times = log_times + torch.triu(torch.full((self.n_species,self.n_species), float("Inf")))
            log_times = log_times.detach()
            tree = Tree(self.theta,
                        log_times,
                        self.tree_log_probs.detach(),
                        pop_size=self.pop_size)
            ELBO_hats[i] = (tree.log_p - tree.log_q).detach()
            log_qs[i] = tree.log_q

        # calculate VIMCO estimator
        ELBO_logsumexp = torch.logsumexp(ELBO_hats,0)
        L_hat_K = ELBO_logsumexp - np.log(batch_size)
        w_tilde = torch.exp(ELBO_hats - ELBO_logsumexp)

        ELBO_logsumexp_minus_j = torch.zeros(batch_size)
        ELBO_hats_minus_j = (torch.sum(ELBO_hats) - ELBO_hats)/(batch_size-1)
        for j in range(batch_size):
            ELBO_logsumexp_minus_j[j] = torch.logsumexp(torch.tensor([x if i != j else ELBO_hats_minus_j[j] for i,x in enumerate(ELBO_hats)]),0) - np.log(batch_size)

        L_hat_K_minus_j = L_hat_K - ELBO_logsumexp_minus_j

        weights = (L_hat_K_minus_j - w_tilde)

        return torch.sum(weights*log_qs)


    def record_metrics(self,iter,test_batch_size):

        self.epochs.append(iter)
        self.run_time += time.time()
        self.run_times.append(self.run_time)

        grad_norm = np.sqrt(np.sum(self.theta.grad.detach().numpy()**2))
        self.grad_norms.append(grad_norm)

        # estimate ELBO
        Zs = torch.normal(mean=0.0,std=1.0,size=(test_batch_size,self.n_species,self.n_species))
        ELBO = self.ELBO_reparam(test_batch_size).detach()
        self.ELBO_ests.append(ELBO)

        print("iteration: ",iter)
        print("runtime: %d mins"% np.floor(np.sum(self.run_times) / 60))
        print("step size: ", self.optimizer.param_groups[0]['lr'])
        print("grad_norm estimate: ", grad_norm)
        print("ELBO estimate: ", ELBO.item())
        print("")

        self.run_time = -time.time()

        return

    def learn(self,batch_size,iters,alpha,method="reparam",
              anneal_freq=1,anneal_rate=1.0,record_every=10,test_batch_size=100,
              pop_size=1.0,max_time=12.0,linear_decay=False):

        # initialize the optimizer
        self.optimizer = torch.optim.Adam([self.theta], lr=alpha)
        self.run_time = -time.time()

        for it in range(iters):

            if method == 'reparam':
                loss = -self.ELBO_reparam(batch_size)
            elif method == 'reinforce':
                loss = -self.ELBO_reinforce(batch_size)
            elif method == 'VIMCO':
                loss = -self.ELBO_VIMCO(batch_size)
            else:
                raise NotImplementedError

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if it % record_every == 0:
                self.record_metrics(it,test_batch_size)
            if linear_decay:
                self.optimizer.param_groups[0]['lr'] = alpha*(1-it/iters)
            elif it % anneal_freq == 0:
                self.optimizer.param_groups[0]['lr'] *= anneal_rate

            if (np.sum(self.run_times) / 3600) > max_time:
                print("Maximum time exceeded. Exiting...")
                return

        return

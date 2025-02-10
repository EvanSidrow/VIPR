import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

import time
import math
import random
import numpy as np
import datetime

from scipy.cluster.hierarchy import linkage

from tree_torch import Tree

class SLCVI(nn.Module):

    def __init__(self,genomes,theta,pop_size):
        super().__init__()

        # set parameters
        self.theta = nn.Parameter(torch.tensor(theta, requires_grad=True))

        # set values for tree
        self.nuc2vec = {'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
                        '-':[1.,1.,1.,1.], '?':[1.,1.,1.,1.], 'N':[1.,1.,1.,1.], 'R':[1.,1.,0.,0.],
                        'Y':[0.,0.,1.,1.], 'S':[0.,1.,1.,0.], 'W':[1.,0.,0.,1.], 'K':[0.,1.,0.,1.],
                        'M':[1.,0.,1.,0.], 'B':[0.,1.,1.,1.], 'D':[1.,1.,0.,1.], 'H':[1.,0.,1.,1.],
                        'V':[1.,1.,1.,0.], '.':[1.,1.,1.,1.], 'U':[0.,0.,0.,1.], 'n':[1.,1.,1.,1.]}
        for key in list(nuc2vec):
            self.nuc2vec[key.lower()] = self.nuc2vec[key]

        self.ntaxa = data.shape[0]
        self.pop_size = pop_size
        self.L, self.site_counts = map(torch.FloatTensor, self.initialCLV(genomes, unique_site=True))
        self.D, self.U, self.U_inv, self.rateM = self.decompJC()

        # values from optimization
        self.optimize = None
        self.ELBO_ests = []
        self.multi_ELBO_ests = []
        self.grad_norms = []
        self.run_time = None
        self.run_times = []
        self.epochs = []

    def decompJC(self,symm=False):
        # pA = pG = pC = pT = .25
        pden = np.array([.25, .25, .25, .25])
        rate_matrix_JC = 1.0/3 * np.ones((4,4))
        for i in range(4):
            rate_matrix_JC[i,i] = -1.0

        if not symm:
            D_JC, U_JC = np.linalg.eig(rate_matrix_JC)
            U_JC_inv = np.linalg.inv(U_JC)
        else:
            D_JC, W_JC = np.linalg.eigh(np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_JC), np.diag(np.sqrt(1.0/pden))))
            U_JC = np.dot(np.diag(np.sqrt(1.0/pden)), W_JC)
            U_JC_inv = np.dot(W_JC.T, np.diag(np.sqrt(pden)))

        return D_JC, U_JC, U_JC_inv, rate_matrix_JC

    def initialCLV(self, genomes, unique_site=False):
        if unique_site:
            data_arr = np.array([list(genome) for genome in genomes]).T
            unique_sites, counts = np.unique(data_arr, return_counts=True, axis=0)
            n_unique_sites = len(counts)
            unique_data = unique_sites.T
            return [np.transpose([self.nuc2vec[c] for c in unique_data[i]]) for i in range(self.ntaxa)], counts
        else:
            return [np.transpose([self.nuc2vec[c] for c in data[i]]) for i in range(self.ntaxa)]

    def scipy_linkage_to_ete3(self,linkage_matrix):
        """
        Converts a SciPy hierarchical clustering linkage matrix to an ETE3 tree.

        Parameters:
            linkage_matrix (ndarray): The linkage matrix from scipy.cluster.hierarchy.linkage.
            labels (list of str): Labels for the original leaves (samples).

        Returns:
            ete3.Tree: The converted ETE3 tree.
            branches: tensor of branch lengths
        """
        nodes = {i: Tree(name=i,dist=0.0) for i in range(self.ntaxa)}  # Initialize leaf nodes
        branches = torch.zeros(2*self.ntaxa-2) # Initialize branches

        for i, (left, right, distance, _) in enumerate(linkage_matrix):
            left, right = int(left), int(right)  # Convert indices to integers
            new_node = Tree(name=num_leaves+i)  # Create an internal node
            new_node.dist = distance  # Set branch length
            new_node.add_child(nodes[left])  # Add left child
            new_node.add_child(nodes[right])  # Add right child
            for child in new_node.children:
                child.branch = distance - child.dist
                branches[child.name] = child.branch

            nodes[num_leaves + i] = new_node  # Store the new node in the dictionary

        tree = nodes[2*num_leaves-1]
        tree.branch = None
        return tree, branches

    def multisample_ELBO_reparam(self,batch_size):

        ELBO_hats = torch.zeros(batch_size)
        for i in range(batch_size):
            Z = torch.normal(mean=0.0,std=1.0,size=(self.ntaxa,self.ntaxa))
            log_times = torch.exp(self.theta[1])*Z+self.theta[0]
            log_times = log_times + torch.triu(torch.full((self.ntaxa,self.ntaxa), float("Inf")))
            tree = Tree(self.theta,
                        log_times,
                        self.tree_log_probs.detach(),
                        pop_size=self.pop_size)
            ELBO_hats[i] = tree.log_p - tree.log_q
        return torch.logsumexp(ELBO_hats,0) - np.log(batch_size)

    def ELBO_reparam(self,batch_size):

        ELBO = 0
        for _ in range(batch_size):
            Z = torch.normal(mean=0.0,std=1.0,size=(self.ntaxa,self.ntaxa))
            log_times = torch.exp(self.theta[1])*Z+self.theta[0]
            log_times = log_times + torch.triu(torch.full((self.ntaxa,self.ntaxa), float("Inf")))
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
            Z = torch.normal(mean=0.0,std=1.0,size=(self.ntaxa,self.ntaxa))
            log_times = torch.exp(self.theta[1])*Z+self.theta[0]
            log_times = log_times + torch.triu(torch.full((self.ntaxa,self.ntaxa), float("Inf")))
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
            Z = torch.normal(mean=0.0,std=1.0,size=(self.ntaxa,self.ntaxa))
            log_times = torch.exp(self.theta[1])*Z+self.theta[0]
            log_times = log_times + torch.triu(torch.full((self.ntaxa,self.ntaxa), float("Inf")))
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
        with torch.no_grad():

            ELBO = self.ELBO_reparam(test_batch_size)
            self.ELBO_ests.append(ELBO)

            # estimate 10 sample ELBO
            multi_ELBO = self.multisample_ELBO_reparam(test_batch_size)
            self.multi_ELBO_ests.append(multi_ELBO)

        print("iteration: ",iter)
        print("runtime: %d mins"% np.floor(np.sum(self.run_times) / 60))
        print("step size: ", self.optimizer.param_groups[0]['lr'])
        print("grad_norm estimate: ", grad_norm)
        print("ELBO estimate: ", ELBO.item())
        print("%s-sample ELBO estimate: "%test_batch_size, multi_ELBO.item())
        print("")

        self.run_time = -time.time()

        return

    def ELBO_reparam_scipy(batch_size):

        ELBO = 0
        for _ in range(batch_size):
            Z = torch.normal(mean=0.0,std=1.0,size=(self.ntaxa,self.ntaxa))
            log_times = torch.exp(self.theta[1])*Z+self.theta[0]
            tree = linkage([log_times[ind[0],ind[1]] for ind in combinations(range(4),2)], method="single")

    def learn(self,batch_size,iters,alpha,method="reparam",
              anneal_freq=1,anneal_rate=1.0,record_every=100,test_batch_size=500,
              pop_size=1.0,max_time=12.0,linear_decay=False):

        # initialize the optimizer
        self.optimizer = torch.optim.Adam([self.theta], lr=alpha)
        self.run_time = -time.time()

        for it in range(iters):

            eps = torch.normal(mean=0.0,std=1.0,size=(n_species,n_species))
            log_times = torch.exp(theta[1])*eps+theta[0]
            log_times = log_times + torch.triu(torch.full((n_species,n_species), float("Inf")))
            log_times0 = log_times.detach().numpy()
            Z = linkage([np.exp(log_times0[ind[1],ind[0]]) for ind in combinations(range(n_species),2)], method="single")
            Z = torch.tensor(Z)

            # find the right inds
            rowinds,colinds = np.where(np.isin(np.exp(log_times0),Z[:,2]))
            Z[:,2] = torch.exp(torch.sort(log_times[rowinds,colinds])[0])

            tree,branches = self.scipy_linkage_to_ete3(Z)
            loss = -self.loglikelihood(tree,branches)

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

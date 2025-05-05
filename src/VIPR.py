import torch
import torch.nn as nn
from torch.distributions.log_normal import LogNormal
from torch.distributions.exponential import Exponential
from torch.distributions.categorical import Categorical
torch.set_default_dtype(torch.float32)

import time
import numpy as np

from ete3 import Tree

from scipy.cluster.hierarchy import linkage
from itertools import combinations
from itertools import permutations

class VIPR(nn.Module):

    def __init__(self,genomes,means,vars,var_dist="LogNormal",
                 phi_pop_size=None,var_dist_pop_size="Fixed",
                 theta_pop_size=None,prior_pop_size="Fixed",
                 tip_dates=None,
                 phi_rate=None,var_dist_rate="Fixed",
                 theta_rate=None,prior_rate="Fixed"):
        
        super().__init__()

        # validate variational distributions with priors
        if prior_pop_size == "Fixed" and var_dist_pop_size == "LogNormal":
            raise ValueError("Cannot use Fixed prior without Fixed variational distribution")
        if prior_rate == "Fixed" and var_dist_rate == "LogNormal":
            raise ValueError("Cannot use Fixed prior without Fixed variational distribution")

        # set basic values
        self.ntaxa = len(genomes)

        # set variational parameters for popuation size
        self.phi_pop_size = nn.Parameter(phi_pop_size)
        self.var_dist_pop_size = var_dist_pop_size

        # set prior distribution for popuation size
        self.theta_pop_size = theta_pop_size
        self.prior_pop_size = prior_pop_size

        # set variational parameters for rates
        self.phi_rate = nn.Parameter(phi_rate)
        self.var_dist_rate = var_dist_rate

        # set prior distribution for rates
        self.theta_rate = theta_rate
        self.prior_rate = prior_rate

        # set tip dates
        if tip_dates is None:
            self.tip_dates = torch.zeros(self.ntaxa)
        else:
            self.tip_dates = tip_dates

        self.tip_matrix = torch.zeros(self.ntaxa,self.ntaxa)
        for i,j in permutations(range(self.ntaxa),2):
            self.tip_matrix[i,j] = max(self.tip_dates[i],self.tip_dates[j])

        # set parameters from empirical means and vars of branch lengths
        if var_dist == "LogNormal":
            mus = torch.tril(torch.log(means**2 / torch.sqrt(vars + means**2)),diagonal=-1)
            log_sigs = torch.tril(torch.log(torch.sqrt(torch.log(1 + vars / means**2))),diagonal=-1)
            self.phi = nn.Parameter(torch.stack([mus,log_sigs]))
        elif var_dist == "Exponential":
            log_lambs = torch.tril(-torch.log(means),diagonal=-1)
            self.phi = nn.Parameter(log_lambs.unsqueeze(0))
        elif var_dist == "Mixture":
            logit_pis = 0.0*torch.ones(self.ntaxa,self.ntaxa)
            mus = torch.tril(torch.log(means**2 / torch.sqrt(vars + means**2)),diagonal=-1)
            log_sigs = torch.tril(torch.log(torch.sqrt(torch.log(1 + vars / means**2))),diagonal=-1)
            log_lambs = torch.tril(-torch.log(means),diagonal=-1)
            self.phi = nn.Parameter(torch.stack([logit_pis,mus,log_sigs,log_lambs]))
        else:
            raise NotImplementedError
        self.var_dist = var_dist

        # set values for tree
        self.nuc2vec = {'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
                        '-':[1.,1.,1.,1.], '?':[1.,1.,1.,1.], 'N':[1.,1.,1.,1.], 'R':[1.,1.,0.,0.],
                        'Y':[0.,0.,1.,1.], 'S':[0.,1.,1.,0.], 'W':[1.,0.,0.,1.], 'K':[0.,1.,0.,1.],
                        'M':[1.,0.,1.,0.], 'B':[0.,1.,1.,1.], 'D':[1.,1.,0.,1.], 'H':[1.,0.,1.,1.],
                        'V':[1.,1.,1.,0.], '.':[1.,1.,1.,1.], 'U':[0.,0.,0.,1.], 'n':[1.,1.,1.,1.]}

        for key in list(self.nuc2vec):
            self.nuc2vec[key.lower()] = self.nuc2vec[key]

        self.L, self.site_counts = map(torch.FloatTensor, self.initialCLV(genomes, unique_site=True))
        self.pden = torch.from_numpy(np.array([0.25,0.25,0.25,0.25])).float()

        # values for optimization
        self.optimize = None
        self.ELBO_ests = []
        self.multi_ELBO_ests = []
        self.grad_norms = []
        self.run_time = None
        self.run_times = []
        self.epochs = []
        self.phis = []

    def initialCLV(self, genomes, unique_site=False):

        if unique_site:
            data_arr = np.array([list(genome) for genome in genomes]).T
            unique_sites, counts = np.unique(data_arr, return_counts=True, axis=0)
            unique_data = unique_sites.T
            return [np.transpose([self.nuc2vec[c] for c in unique_data[i]]) for i in range(self.ntaxa)], counts

        else:
            return [np.transpose([self.nuc2vec[c] for c in genomes[i]]) for i in range(self.ntaxa)]

    def logprior(self,linkage_matrix,pop_size):

        # Combine and sort all time points (tips + coalescent times)
        times = torch.concatenate([self.tip_dates, linkage_matrix[:,2]])
        events = torch.concatenate([torch.ones(self.ntaxa), -torch.ones(self.ntaxa-1)])  # 1: tip, -1: coalescent

        # Sort by time
        sorted_indices = np.argsort(times)
        times = times[sorted_indices]
        events = events[sorted_indices]

        ks = torch.cat([torch.tensor([0]),torch.cumsum(events,0)[:-1]])
        ts = torch.diff(times,prepend=torch.tensor([0.0]))

        # only include interval times with at least 2 taxa
        ts = ts[ks > 1]
        events = events[ks > 1]
        ks = ks[ks > 1]

        # calculate rates
        rates = ks*(ks-1) / (2*pop_size)

        # determine which events are coalescent and which are tips
        coal_inds = torch.where(events == -1)[0]
        tip_inds = torch.where(events == 1)[0]

        # add coalescent times (prob that exponential r.v. is equal to coalescent time)
        log_prior = torch.sum(torch.log(rates[coal_inds]) - rates[coal_inds]*ts[coal_inds])

        # add specific tree structure (1/number of ways to pick 2 taxa from k taxa)
        log_prior -= torch.sum(torch.log(ks[coal_inds]*(ks[coal_inds]-1)/2))

        # add tip times (prob that exponential r.v. is greater than tip time)
        log_prior -= torch.sum(rates[tip_inds]*ts[tip_inds])

        return log_prior

        #ts = torch.diff(linkage_matrix[:,2],prepend=torch.tensor([0.0]))
        #ks = torch.tensor(range(self.ntaxa,1,-1))
        #rates = ks*(ks-1) / (2*pop_size)
        #return torch.sum(torch.log(rates) - rates*ts - torch.log(ks*(ks-1)/2))
    
    def logprior_pop_size(self,pop_size):
        if self.prior_pop_size == "Fixed":
            return torch.tensor(0.0)
        elif self.prior_pop_size == "LogNormal":
            return LogNormal(self.theta_pop_size[0], torch.exp(self.theta_pop_size[1])).log_prob(pop_size)
        else:
            raise NotImplementedError
        
    def logprior_rate(self,rate):
        if self.var_dist_rate == "Fixed":
            return torch.tensor(0.0)
        elif self.var_dist_rate == "LogNormal":
            return LogNormal(self.theta_rate[0], torch.exp(self.theta_rate[1])).log_prob(rate)
        else:
            raise NotImplementedError

    def loglikelihood(self,linkage_matrix,rate):

        # convert linkage_matrix to tree and branches
        tree,branches = self.scipy_linkage_to_ete3(linkage_matrix)

        transition_matrix = -torch.expm1(-4.0 * branches * rate / 3.0).unsqueeze(-1).unsqueeze(-1) * 0.25 + \
                               torch.exp(-4.0 * branches * rate / 3.0).unsqueeze(-1).unsqueeze(-1) * torch.eye(4)
        
        scaler_list = []
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node.state = self.L[node.name].detach()
            else:
                node.state = 1.0
                for child in node.children:
                    node.state *= transition_matrix[child.name].mm(child.state)
                scaler = torch.sum(node.state, 0)
                node.state /= scaler
                scaler_list.append(scaler)

        scaler_list.append(torch.mm(self.pden.view(-1,4), tree.state).squeeze())
        logll = torch.sum(torch.stack(scaler_list).log() * self.site_counts)
        return logll

    def scipy_linkage_to_ete3(self,linkage_matrix):

        nodes = {i: Tree(name=i,dist=0.0) for i in range(self.ntaxa)}  # Initialize leaf nodes

        # add torch dist
        for key in nodes:
            nodes[key].dist_torch = torch.tensor(0.0)

        branches = torch.zeros(2*self.ntaxa-2) # Initialize branches

        for i, (left, right, distance, _) in enumerate(linkage_matrix):
            left, right = int(left), int(right)  # Convert indices to integers
            new_node = Tree(name=self.ntaxa+i)  # Create an internal node
            new_node.dist_torch = distance  # Set branch length
            new_node.add_child(nodes[left])  # Add left child
            new_node.add_child(nodes[right])  # Add right child
            for child in new_node.children:
                child.branch = distance - child.dist_torch
                branches[child.name] = child.branch

            nodes[self.ntaxa + i] = new_node  # Store the new node in the dictionary

        nodes[len(linkage_matrix) + self.ntaxa - 1].branch = None
        tree = nodes[len(linkage_matrix) + self.ntaxa - 1]
        return tree, branches

    def get_var_dist(self,phi):

        if self.var_dist == "LogNormal":
            return LogNormal(phi[0], torch.exp(phi[1]))
        elif self.var_dist == "Exponential":
            return Exponential(torch.exp(phi[0]))
        elif self.var_dist == "Mixture":
            pis = torch.softmax(torch.stack([phi[0],
                                             torch.zeros_like(phi[0])],
                                            dim=-1),-1)
            mus = phi[1]
            sigs = torch.exp(phi[2])
            lambs = torch.exp(phi[3])
            return [Categorical(pis),LogNormal(mus,sigs),Exponential(lambs)]
        else:
            raise NotImplementedError

    def logvariational(self,linkage_matrix):

        log_q = 0.0
        X = [torch.tensor([i]) for i in range(self.ntaxa)]

        for row in linkage_matrix:

            Wn = X[int(row[0])]
            Zn = X[int(row[1])]
            t = row[2]

            Wn_grid, Zn_grid = torch.meshgrid(Wn, Zn, indexing='ij')
            rowinds = torch.max(Wn_grid.flatten(), Zn_grid.flatten())
            colinds = torch.min(Wn_grid.flatten(), Zn_grid.flatten())
            dist = self.get_var_dist(self.phi[:,rowinds,colinds])
            t_tips = t - self.tip_matrix[rowinds,colinds]

            if self.var_dist == "Mixture":
                pis = torch.softmax(torch.stack([self.phi[0,rowinds,colinds],
                                    torch.zeros_like(self.phi[0,rowinds,colinds])],
                                    dim=-1),-1)
                log_sfs = torch.log(pis[:,0]*(1.0 - dist[1].cdf(t_tips)) + pis[:,1]*(1.0 - dist[2].cdf(t_tips)))
                log_pdfs = torch.logsumexp(torch.stack([torch.log(pis[:,0]) + dist[1].log_prob(t_tips),
                                                        torch.log(pis[:,1]) + dist[2].log_prob(t_tips)]),0)

            else:
                log_sfs = torch.log(1.0 - dist.cdf(t_tips))
                log_pdfs = dist.log_prob(t_tips)

            log_q += torch.sum(log_sfs) + torch.logsumexp(log_pdfs - log_sfs,0)

            X.append(torch.cat([Wn,Zn]))

        return log_q
    
    def logvariational_pop_size(self,pop_size):
        if self.var_dist_pop_size == "Fixed":
            return torch.tensor(0.0)
        elif self.var_dist_pop_size == "LogNormal":
            return LogNormal(self.phi_pop_size[0], torch.exp(self.phi_pop_size[1])).log_prob(pop_size)
        else:
            raise NotImplementedError
        
    def logvariational_rate(self,rate):
        if self.var_dist_rate == "Fixed":
            return torch.tensor(0.0)
        elif self.var_dist_rate == "LogNormal":
            return LogNormal(self.phi_rate[0], torch.exp(self.phi_rate[1])).log_prob(rate)
        else:
            raise NotImplementedError

    def sample_q(self,detach=True):

        dist = self.get_var_dist(self.phi)

        if type(dist) == list:
            if detach:
                inds = dist[0].sample()
                times = dist[1].sample()
                times[inds == 1] = dist[2].sample()[inds == 1]
            else:
                print("cannot currently do reparam trick with mixture")
                raise NotImplementedError
        else:
            if detach:
                times = dist.sample()
            else:
                times = dist.rsample()

        times = times + self.tip_matrix
        times = times + torch.triu(torch.full((self.ntaxa,self.ntaxa), float("Inf")))
        times0 = times.detach().numpy()

        # perform single linkage clustering
        try:
            Z = linkage([times0[ind[1],ind[0]] for ind in combinations(range(self.ntaxa),2)], 
                         method="single")
        except:
            print([times0[ind[1],ind[0]] for ind in combinations(range(self.ntaxa),2)])
            raise ValueError("Error in linkage calculation")
        
        # populate times with pytorch tensors to pass gradients
        Z = torch.tensor(Z)
        if not detach:
            rowinds,colinds = np.where(np.isin(times0,Z[:,2]))
            new_Z2 = torch.sort(times[rowinds,colinds])[0]

            # check if Z2 is too short
            if len(new_Z2) < len(Z[:,2]):
                rowinds,colinds = np.where(np.isclose(times0,Z[:,2]))
                new_Z2 = torch.sort(times[rowinds,colinds])[0]

            # check if Z2 is too long
            if len(new_Z2) > len(Z[:,2]):
                first_mask = [True] + [(new_Z2[i+1] != new_Z2[i]).item() for i in range(len(new_Z2)-1)]
                new_Z2 = new_Z2[torch.tensor(first_mask)]

            Z[:,2] = new_Z2              

        return Z
    
    def sample_q_pop_size(self,detach=True):
        if self.var_dist_pop_size == "Fixed":
            return self.phi_pop_size[0]
        elif self.var_dist_pop_size == "LogNormal":
            if detach:
                return LogNormal(self.phi_pop_size[0], torch.exp(self.phi_pop_size[1])).sample()
            else:
                return LogNormal(self.phi_pop_size[0], torch.exp(self.phi_pop_size[1])).rsample()
        else:
            raise NotImplementedError
        
    def sample_q_rate(self,detach=True):
        if self.var_dist_rate == "Fixed":
            return self.phi_rate[0]
        elif self.var_dist_rate == "LogNormal":
            if detach:
                return LogNormal(self.phi_rate[0], torch.exp(self.phi_rate[1])).sample()
            else:
                return LogNormal(self.phi_rate[0], torch.exp(self.phi_rate[1])).rsample()
        else:
            raise NotImplementedError

    def ELBO_hats(self,batch_size,detach=False):

        # create VIMCO estimator
        ELBO_hats = torch.zeros(batch_size)
        log_qs = torch.zeros(batch_size)

        # run through batches
        for i in range(batch_size):

            # sample from variational distribution
            pop_size = self.sample_q_pop_size(detach=detach)
            rate = self.sample_q_rate(detach=detach)
            Z = self.sample_q(detach=detach)

            # evaluate joint distribution
            log_prior = self.logprior(Z,pop_size) + self.logprior_pop_size(pop_size) + self.logprior_rate(rate)
            log_ll = self.loglikelihood(Z,rate)

            # evaluate variational distribution
            log_q = self.logvariational(Z) + self.logvariational_pop_size(pop_size) + self.logvariational_rate(rate)

            # store ELBO
            ELBO_hats[i] = (log_ll + log_prior - log_q).detach()
            log_qs[i] = log_q

        return ELBO_hats, log_qs

    def multisample_ELBO_reparam(self,batch_size,detach=False):

        ELBO_hats = self.ELBO_hats(batch_size,detach=detach)[0]
        return torch.logsumexp(ELBO_hats,0) - np.log(batch_size)

    def ELBO_reparam(self,batch_size,detach=False):

        ELBO_hats = self.ELBO_hats(batch_size,detach=detach)[0]
        return torch.mean(ELBO_hats)      

    def ELBO_reinforce(self,batch_size):

        ELBO_hats,log_qs = self.ELBO_hats(batch_size,detach=True)
        weights = ELBO_hats - torch.mean(ELBO_hats)
        return torch.sum(weights*log_qs) / (batch_size-1)

    def ELBO_VIMCO(self,batch_size):

        ELBO_hats,log_qs = self.ELBO_hats(batch_size,detach=True)

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

        grad_norm = np.sqrt(np.sum(self.phi.grad.detach().numpy()**2))
        self.grad_norms.append(grad_norm)

        self.phis.append(self.phi.detach().numpy())

        # estimate ELBO
        with torch.no_grad():

            ELBO = self.ELBO_reparam(test_batch_size,detach=True)
            self.ELBO_ests.append(ELBO)

            # estimate multi-sample ELBO
            multi_ELBO = self.multisample_ELBO_reparam(test_batch_size,detach=True)
            self.multi_ELBO_ests.append(multi_ELBO)

        print("iteration %d:(%.1fs)"%(iter,self.run_time))
        print("step size: ", self.optimizer.param_groups[0]['lr'])
        print("grad_norm estimate: ", grad_norm)
        print("ELBO estimate: ", ELBO.item())
        print("%s-sample ELBO estimate: "%test_batch_size, multi_ELBO.item())
        print("")

        self.run_time = -time.time()

        return

    def learn(self,batch_size,iters,alpha,method="reparam",
              anneal_freq=1,anneal_rate=1.0,record_every=100,test_batch_size=500,
              pop_size=None,max_time=12.0,linear_decay=False):

        if not pop_size is None:
            self.pop_size = pop_size

        # initialize the optimizer
        self.optimizer = torch.optim.Adam([self.phi,
                                           self.phi_pop_size,
                                           self.phi_rate], lr=alpha)
        self.run_time = -time.time()

        for it in range(iters):

            if method == 'reparam':
                loss = -self.ELBO_reparam(batch_size)
            elif method == 'reinforce':
                loss = -self.ELBO_reinforce(batch_size)
            elif method == 'VIMCO':
                loss = -self.ELBO_VIMCO(batch_size)
            elif method == "reinforce_VIMCO":
                current_time = np.sum(self.run_times) / 3600
                if (current_time < max_time/2) and (it < iters/2):
                    loss = -self.ELBO_reinforce(batch_size)
                else:
                    loss = -self.ELBO_VIMCO(batch_size)
            else:
                raise NotImplementedError

            self.optimizer.zero_grad()
            loss.backward()

            if torch.any(torch.isnan(self.phi.grad)):
                print("NaN in gradient. Exiting...")
                return
            
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
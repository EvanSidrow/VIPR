import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

import numpy as np

class Node:

    def __init__(self,name,log_probs):

        self.name = name
        self.leaves = set([])
        self.parent = None
        self.children = []
        self.coal_time = 0.0
        self.log_probs = log_probs


class Tree:

    def __init__(self,theta,log_times,tree_log_probs,pop_size=1.0):

        # basic properties
        self.theta = theta
        self.log_times = log_times
        self.ntaxa = tree_log_probs.shape[0]
        self.ngenes = tree_log_probs.shape[1]
        self.pop_size = pop_size

        # distribution values
        self.log_prior = 0.0
        self.log_like = 0.0
        self.log_p = 0.0
        self.log_q = 0.0
        self.log_stationary_dist = np.log(0.25)*torch.ones(4,dtype=torch.float64)

        # initialize leaves
        self.leaves = []
        for leaf_num in range(self.ntaxa):
            self.leaves.append(Node(leaf_num,tree_log_probs[leaf_num]))
            self.leaves[-1].leaves.add(leaf_num)

        # initialize nodes
        self.nodes = []
        for node_num in range(self.ntaxa,2*self.ntaxa-1):
            self.nodes.append(Node(node_num,torch.zeros((self.ngenes,4))))

        # initialze all coal_times
        self.coal_times = torch.zeros(self.ntaxa)

        # draw graph
        self.draw_graph()

    def logdotexp(self, A, B):
        max_A = torch.max(A)
        max_B = torch.max(B)
        C = torch.matmul(torch.exp(A - max_A), torch.exp(B - max_B))
        C = torch.log(C)
        C = C + max_A + max_B
        return C

    def find_ancestors(self,leaf_nums_to_join):

        ancestors = []
        for leaf_num in leaf_nums_to_join:
            ancestor = self.leaves[leaf_num]
            while ancestor.parent is not None:
                ancestor = ancestor.parent
            ancestors.append(ancestor)

        return ancestors


    def update_q(self,ancestors):

        t = ancestors[0].parent.coal_time
        log_t = torch.log(t)

        log_cum_sfs = 0.0
        log_pdfs_over_sfs = torch.zeros(len(ancestors[0].leaves),
                                        len(ancestors[1].leaves))

        for a,leaf_a in enumerate(ancestors[0].leaves):
            for b,leaf_b in enumerate(ancestors[1].leaves):

                leaf_i = max(leaf_a,leaf_b)
                leaf_j = min(leaf_a,leaf_b)

                # update q
                mu = self.theta[0,leaf_i,leaf_j]
                log_sig = self.theta[1,leaf_i,leaf_j]
                sig = torch.exp(log_sig)

                log_sf = torch.log(0.5 - 0.5*torch.erf((log_t-mu)/(np.sqrt(2)*sig)))
                log_pdf = -log_sig-0.5*np.log(2*np.pi)-log_t
                log_pdf = log_pdf - (((log_t-mu)/sig)**2)/2

                log_cum_sfs = log_cum_sfs + log_sf
                log_pdfs_over_sfs[a,b] = log_pdf - log_sf

        # update q
        self.log_q = self.log_q + log_cum_sfs + torch.logsumexp(torch.flatten(log_pdfs_over_sfs),0)

        return


    def draw_graph(self,log_times=None,theta=None,pop_size=None):

        if log_times is None:
            log_times = self.log_times
        if theta is None:
            theta = self.theta
        if pop_size is None:
            pop_size = self.pop_size

        node_leaf_nums = [[i] for i in range(self.ntaxa)]
        ravel_log_times = torch.ravel(log_times)
        node_num = 0
        prev_coal_time = 0 # for prior update

        for ind in torch.argsort(ravel_log_times):

            if ind == 0:
                break

            leave_nums_to_join = torch.unravel_index(ind,log_times.shape)

            # find ancestors of leaves to join
            ancestors = self.find_ancestors(leave_nums_to_join)

            # find if leaves already joined:
            leaves_joined = (ancestors[0] == ancestors[1])

            if not leaves_joined:

                # get coal time from log
                coal_time = torch.exp(ravel_log_times[ind])

                # grab a node to define as a parent
                ancestor_parent = self.nodes[node_num]
                ancestor_parent.coal_time = coal_time
                ancestor_parent.leaves = ancestors[0].leaves.union(ancestors[1].leaves)

                for ancestor in ancestors:

                    # set parent and children
                    ancestor.parent = ancestor_parent
                    ancestor_parent.children.append(ancestor)

                    # set the edge lengths
                    edge_length = coal_time-ancestor.coal_time

                    # evaluate likelihood of transition
                    P = 0.25*(1-torch.exp(-4*edge_length/3))*torch.ones((4,4),dtype=torch.float64)
                    P = P + torch.eye(4)*(torch.exp(-4*edge_length/3))
                    ancestor_parent.log_probs = ancestor_parent.log_probs + self.logdotexp(ancestor.log_probs,torch.log(P))

                # update q
                self.update_q(ancestors)

                # update prior
                k = self.ntaxa - node_num
                rate = k*(k-1) / (2*self.pop_size)
                self.log_prior = self.log_prior + np.log(rate) - rate*(coal_time-prev_coal_time)
                self.log_prior = self.log_prior - np.log(k*(k-1)/2)
                prev_coal_time = coal_time

                # update ancestor numbers
                node_num = node_num + 1

        # put in log likelihood
        self.log_like = torch.sum(self.logdotexp(self.nodes[-1].log_probs,self.log_stationary_dist))
        self.log_p = self.log_like + self.log_prior

        return

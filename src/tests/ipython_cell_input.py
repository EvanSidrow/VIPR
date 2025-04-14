
import matplotlib.pyplot as plt
import numpy as np

import torch

from itertools import product
from itertools import combinations
import pickle
import datetime

import subprocess
import os

from scipy.special import logsumexp

from Bio.Phylo.TreeConstruction import _DistanceMatrix
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceCalculator
from io import StringIO
from Bio import Phylo

from ete3 import Tree
import torch.nn as nn
from torch.distributions.log_normal import LogNormal

import warnings

warnings.filterwarnings('ignore')

from copy import deepcopy
from scipy.cluster.hierarchy import linkage,dendrogram
from scipy.cluster.hierarchy import *


### set parameters

np.random.seed(0)

data_set = "DS1"
pop_size = 5.0 # exponential parameter for constant pop size prior
data_file = '../../dat/'+data_set+'/'+data_set+'.pickle'
time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
reparam_file = '../../results/'+data_set+'/'+data_set+'_'+time+'_reparam.pickle'
reinforce_file = '../../results/'+data_set+'/'+data_set+'_'+time+'_reinforce.pickle'
VIMCO_file = '../../results/'+data_set+'/'+data_set+'_'+time+'_VIMCO.pickle'

### load in data

with open(data_file, 'rb') as f:
    DS1 = pickle.load(f)
    
genomes = []
species = []
for key in DS1:
    genomes.append(DS1[key])
    species.append(key)
ntaxa = len(species)

nuc2vec = {'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
           '-':[1.,1.,1.,1.], '?':[1.,1.,1.,1.], 'N':[1.,1.,1.,1.], 'R':[1.,1.,0.,0.],
           'Y':[0.,0.,1.,1.], 'S':[0.,1.,1.,0.], 'W':[1.,0.,0.,1.], 'K':[0.,1.,0.,1.],
           'M':[1.,0.,1.,0.], 'B':[0.,1.,1.,1.], 'D':[1.,1.,0.,1.], 'H':[1.,0.,1.,1.],
           'V':[1.,1.,1.,0.], '.':[1.,1.,1.,1.], 'U':[0.,0.,0.,1.], 'n':[1.,1.,1.,1.]}

treedata = ""
ntrees = 0
for i in range(1):
    tree_file = "../../dat/"+data_set+"/"+data_set+"_fixed_pop_support_short_run_rep_%d.trees"%(i+1)
    with open(tree_file, "r") as file:
        for line in file:
            if line.startswith("tree STATE"):
                line = line[line.find('('):]
                line = line.replace("[&rate=1.0]","")
                treedata = f'{treedata}{line}\n'
                #treedata = treedata + line + "\n"
                ntrees += 1
                
            if ntrees > 100:
                break
                

### initialize theta
theta = torch.zeros((2,ntaxa,ntaxa))
trees = Phylo.parse(StringIO(treedata), "newick")
log_dists = np.zeros((ntrees,ntaxa,ntaxa))

for i,tree in enumerate(trees):
    for j in range(ntaxa):
        for k in range(j):
            log_dists[i,j,k] = np.log(tree.distance(target1=str(j+1),target2=str(k+1))/2.0)
            
    if i > 10:
        break

for j in range(ntaxa):
    for k in range(j):
        theta[0,j,k] = np.mean(log_dists[:,j,k])
        theta[1,j,k] = np.log(np.std(log_dists[:,j,k]))
        
### get the rate matrix

def decompJC(symm=False):
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

D,U,U_inv,rate_matrix = decompJC()

pden = np.array([0.25,0.25,0.25,0.25])
pden = torch.from_numpy(pden).double()
D = torch.from_numpy(D).double()
U = torch.from_numpy(U).double()
U_inv = torch.from_numpy(U_inv).double()

### define needed functions

def initialCLV(genomes, unique_site=False):
    if unique_site:
        data_arr = np.array([list(genome) for genome in genomes]).T
        unique_sites, counts = np.unique(data_arr, return_counts=True, axis=0)
        n_unique_sites = len(counts)
        unique_data = unique_sites.T

        return [np.transpose([nuc2vec[c] for c in unique_data[i]]) for i in range(ntaxa)], counts
    else:
        return [np.transpose([nuc2vec[c] for c in data[i]]) for i in range(ntaxa)]

L, site_counts = map(torch.DoubleTensor, initialCLV(genomes, unique_site=True))
    
def loglikelihood(tree,branch):
        
    branch_D = torch.einsum("i,j->ij", (branch, D))
    transition_matrix = torch.matmul(torch.einsum("ij,kj->kij", (U, torch.exp(branch_D))), U_inv).clamp(0.0)
    scaler_list = []
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.state = L[node.name].detach()
        else:
            node.state = 1.0
            for child in node.children:
                node.state *= transition_matrix[child.name].mm(child.state)
            scaler = torch.sum(node.state, 0)
            node.state /= scaler
            scaler_list.append(scaler)

    scaler_list.append(torch.mm(pden.view(-1,4), tree.state).squeeze())
    logll = torch.sum(torch.stack(scaler_list).log() * site_counts)
    return logll

def scipy_linkage_to_ete3(linkage_matrix):
    """
    Converts a SciPy hierarchical clustering linkage matrix to an ETE3 tree.

    Parameters:
        linkage_matrix (ndarray): The linkage matrix from scipy.cluster.hierarchy.linkage.
        labels (list of str): Labels for the original leaves (samples).
    
    Returns:
        ete3.Tree: The converted ETE3 tree.
        branches: tensor of branch lengths
    """
    num_leaves = len(linkage_matrix) + 1
    nodes = {i: Tree(name=i,dist=0.0) for i in range(num_leaves)}  # Initialize leaf nodes
    # add torch dist
    for key in nodes:
        nodes[key].dist_torch = torch.tensor(0.0)
        
    branches = torch.zeros(2*num_leaves-2) # Initialize branches
    
    for i, (left, right, distance, _) in enumerate(linkage_matrix):
        left, right = int(left), int(right)  # Convert indices to integers
        new_node = Tree(name=num_leaves+i)  # Create an internal node
        new_node.dist_torch = distance  # Set branch length
        new_node.add_child(nodes[left])  # Add left child
        new_node.add_child(nodes[right])  # Add right child
        for child in new_node.children:
            child.branch = distance - child.dist_torch
            branches[child.name] = child.branch
            
        nodes[num_leaves + i] = new_node  # Store the new node in the dictionary

    nodes[len(linkage_matrix) + num_leaves - 1].branch = None
    tree = nodes[len(linkage_matrix) + num_leaves - 1] 
    return tree, branches

def logvariational_new(linkage_matrix,theta):
     
    log_q = 0.0
    X = [torch.tensor([i]) for i in range(ntaxa)]
    
    for row in linkage_matrix:
        
        Wn = X[int(row[0])]
        Zn = X[int(row[1])]
        t = row[2]
        log_t = torch.log(t)
        
        Wn_grid, Zn_grid = torch.meshgrid(Wn,Zn, indexing='ij')
        rowinds = torch.max(Wn_grid.flatten(), Zn_grid.flatten())
        colinds = torch.min(Wn_grid.flatten(), Zn_grid.flatten())
        
        mus = theta[0,rowinds,colinds]
        sigs = torch.exp(theta[1,rowinds,colinds])
        
        dist = LogNormal(mus, sigs)
        
        log_sfs = torch.log(1.0 - dist.cdf(t))
        log_pdfs = dist.log_prob(t)
        
        log_q += torch.sum(log_sfs) + torch.logsumexp(log_pdfs - log_sfs,0)
            
        X.append(torch.cat([Wn,Zn]))
        
    return log_q


### run VI


theta = nn.Parameter(torch.tensor(theta, requires_grad=True))
optimizer = torch.optim.Adam([theta], lr=0.01)
ct = datetime.datetime.now()

niter = 100
batch_size = 10

for it in range(niter):

    logll = 0.0
    logprior = 0.0
    logq = 0.0
    
    for _ in range(batch_size):
    
        eps = torch.normal(mean=0.0,std=1.0,size=(ntaxa,ntaxa))
        times = torch.exp(torch.exp(theta[1])*eps+theta[0])
        times = times + torch.triu(torch.full((ntaxa,ntaxa), float("Inf")))
        times0 = times.detach().numpy()
        Z = linkage([times0[ind[1],ind[0]] for ind in combinations(range(ntaxa),2)], method="single")
        Z = torch.tensor(Z)

        # find the right inds
        rowinds,colinds = np.where(np.isin(times0,Z[:,2]))
        Z[:,2] = torch.sort(times[rowinds,colinds])[0]

        # find log-priors
        ts = torch.diff(Z[:,2],prepend=torch.tensor([0.0]))
        ks = torch.tensor(range(ntaxa,1,-1))
        rates = ks*(ks-1) / (2*pop_size)
        logprior = logprior + torch.sum(torch.log(rates) - rates*ts - torch.log(ks*(ks-1)/2))/batch_size

        # get log-likelihood
        tree,branches = scipy_linkage_to_ete3(Z)
        logll = logll + loglikelihood(tree,branches)/batch_size
        
        # get log-q
        logq = logq + logvariational_new(Z,theta)/batch_size
        
    loss = logq-logll-logprior

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if it % (niter/25) == 0:
        print(it/niter,loss)
    
print(datetime.datetime.now() - ct)
print(theta)

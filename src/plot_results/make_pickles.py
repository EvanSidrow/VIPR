#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import math
import seaborn as sns
import numpy as np
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
sys.path.append('vbpi-torch/rooted')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import torch
from dataManipulation import *
from treeManipulation import *
from utils import tree_summary, summary, summary_raw, get_support_info
from vbpi import VBPI

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_set', type=str, required=True, help=' HCV ')
args = parser.parse_args()


# In[2]:


sns.set_style("whitegrid")  # Options: white, dark, whitegrid, darkgrid, ticks
sns.set_palette("muted")    # Options: deep, muted, bright, pastel, dark, colorblind


# In[3]:


import pandas as pd

#from autograd_gamma import gamma, gammainc, gammaincc, gammaincln, gammainccln

from copy import deepcopy
from itertools import product
from itertools import combinations
import pickle

from io import StringIO
from Bio import Phylo

from tree_torch import Tree
from SLCVI_torch import SLCVI

import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)


# # Plot:
#  - time / likelihood evaluation vs number of taxa
#  - time / likelihood evaluation vs number of sites

# In[4]:

data_set = args.data_set

print(data_set)
print("")

pop_size = 5.0 # exponential parameter for constant pop size prior

# initialize models
models = {"reinforce": {},
          "reparam": {},
          "VIMCO": {},
          "VBPI": {}}

data_file = '../dat/'+data_set+'/'+data_set+'.pickle'

# models
models = ["reparam","reinforce","VIMCO","BEAST","VBPI"]

# Beast file
BEAST_pref = '../dat/'+data_set+'/'+data_set+'_MLL_'
BEAST_burnin = 250000

# VPBI files
VBPI_dir = '../results/'+data_set+'/'


# # Requirements for VBPI

# In[8]:


# load the sequence data and estimate the subsplit support
data, taxa = loadData('../dat/'+data_set+'/'+data_set+'.nexus', 'nexus')
mcmc_support_trees_dict, mcmc_support_trees_wts = summary('../dat/'+data_set+'/'+data_set+'_fixed_pop_support_short_run', 'nexus', burnin=250)
rootsplit_supp_dict, subsplit_supp_dict = get_support_info(taxa, mcmc_support_trees_dict)
#del mcmc_support_trees_dict, mcmc_support_trees_wts


# In[9]:


# load the ground truth
#mcmc_sampled_trees_dict, mcmc_sampled_trees_wts, _ = tree_summary('../dat/DS1/DS1_fixed_pop_golden_run.trees', 'nexus', burnin=25001)
emp_tree_freq = None#{mcmc_sampled_trees_dict[tree_id]: tree_wts for tree_id, tree_wts in sorted(mcmc_sampled_trees_wts.items(), key=lambda x:x[1], reverse=True)}
sample_info = [0.0 for taxon in taxa]
#del mcmc_sampled_trees_dict, mcmc_sampled_trees_wts


# In[10]:


# set up the model
VBPI_models = {}
VBPI_models["10"] = VBPI(taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden=np.ones(4)/4., subModel=('JC', 1.0),
             emp_tree_freq=emp_tree_freq, root_height_offset=0.0, clock_rate=1.0, psp=True,
             sample_info=sample_info, coalescent_type='fixed_pop', clock_type='fixed_rate',
             log_pop_size_offset=math.log(5.0))

VBPI_models["20"] = VBPI(taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden=np.ones(4)/4., subModel=('JC', 1.0),
             emp_tree_freq=emp_tree_freq, root_height_offset=0.0, clock_rate=1.0, psp=True,
             sample_info=sample_info, coalescent_type='fixed_pop', clock_type='fixed_rate',
             log_pop_size_offset=math.log(5.0))


# # Load in the relevant files

# In[11]:


# VBPI
VBPI_runtimes = {}
VBPI_lbs = {}
VBPI_iters = {}

legend = []

for bs in ["10","20"]:

    VBPI_runtimes[bs] = None
    VBPI_lbs[bs] = None
    VBPI_iters[bs] = None
    lb_star = -np.infty
    file_star = None

    for ss in ["0.003","0.001","0.0003","0.0001"]:
    #for ss in ["1e-05"]:
        VBPI_pref = "mcmc_vimco_%s_%s_psp_fixed_pop_fixed_rate_"%(bs,ss)
        files = [x for x in os.listdir(VBPI_dir) if x.startswith(VBPI_pref)]
        files = [x for x in files if x.endswith(".pt")]
        new_files = []
        for x in files:
            if "2025-01-22" in x or "2025-01-23" in x or "2025-01-24" in x or "2025-01-25" in x:

                new_files.append(x)

        files = new_files
        print(len(files))

        for file in files:

            VBPI_runtimes0 = np.load(VBPI_dir+file.replace(".pt","_run_time.npy"))
            VBPI_lbs0 = np.load(VBPI_dir+file.replace(".pt","_test_lb.npy"))
            VBPI_iters0 = np.load(VBPI_dir+file.replace(".pt","_iters.npy"))

            if np.mean(VBPI_lbs0[-10:]) > lb_star:
                VBPI_runtimes[bs] = VBPI_runtimes0
                VBPI_lbs[bs] = VBPI_lbs0
                VBPI_iters[bs] = VBPI_iters0
                VBPI_models[bs].load_from(VBPI_dir+max([x for x in files if x.endswith(".pt")]))
                lb_star = np.mean(VBPI_lbs0[-10:])
                file_star = file


    print(file_star)


# In[12]:


# BEAST
def load_beast(data_set,i,burnin):
    df = pd.read_csv('../dat/'+data_set+'/'+data_set+'_fixed_pop_MLL_%d.log'%i,
                     sep = '\t',skiprows=[0,1,2])
    df = df[df.state > burnin]
    return df

BEAST_data = pd.concat([load_beast(data_set,i,BEAST_burnin) for i in range(1,11)])
BEAST_MLLs = []

# extract MLL from beast log
for i in range(1,11):
    with open('../dat/'+data_set+'/'+data_set+"_MLL_%d.txt"%i, "r") as text_file:
        line = text_file.readlines()[-4]
    ind = np.where([not x in "-1234567890." for x in line])[0][-2]
    BEAST_MLLs.append(float(line[(ind+1):-1]))


# In[13]:


# my models
optims = {}
settings = {}
ELBO_star = -np.infty
fname_star = None
ELBO_min = np.infty
ELBO_max = -np.infty

for model in ["reinforce","reparam","VIMCO"]:

    optims[model] = None
    ELBO_star = -np.infty

    for ss in [0.03,0.01,0.003]:

        for rs in range(10):

            optim_dir = '../results/'+data_set+'/'
            optim_pref = data_set+'_'+model+'_'+str(ss)+'_'+str(rs)
            files = [x for x in os.listdir(optim_dir) if x.startswith(optim_pref)]

            if data_set == "DS14":
                files = [x for x in files if "2025_01_25_00" in x]
            else:
                files = [x for x in files if not "2025_01_23" in x]
                files = [x for x in files if not "2025_01_24" in x]
                files = [x for x in files if not "2025_01_25" in x]

            if not files:
                print(optim_pref + " does not exist. Continuing...")
                continue

            fname = optim_dir + max(files)

            with open(fname, 'rb') as f:
                optim0 = pickle.load(f)

            if np.mean(optim0.multi_ELBO_ests[-10:]) > ELBO_star:
                optims[model] = optim0
                settings[model] = (ss,rs)
                ELBO_star = np.mean(optim0.multi_ELBO_ests[-10:])
                fname_star = fname

            #print(model)
            #print(ss)
            #print(rs)
            #print(max(optim0.multi_ELBO_ests))
            #print("")

    print(fname_star)

    if max(optims[model].multi_ELBO_ests) > ELBO_max:
        ELBO_max = max(optims[model].multi_ELBO_ests)

    if min(optims[model].multi_ELBO_ests) < ELBO_min:
        ELBO_min = min(optims[model].multi_ELBO_ests)


for model in optims:
    optims[model].epochs = [i*10 for i in range(len(optims[model].epochs))]


# # plot estimated ELBO over time

# In[19]:


tree_lengths = {}
root_heights = {}
log_likes = {}
log_priors = {}
p_qs = {}


# # Get data from BEAST

# In[20]:


# get lengths from BEAST
tree_lengths["BEAST"] = BEAST_data.treeLength[BEAST_data.state > BEAST_burnin].to_numpy()
root_heights["BEAST"] = BEAST_data['treeModel.rootHeight'][BEAST_data.state > BEAST_burnin].to_numpy()
log_likes["BEAST"] = BEAST_data.likelihood[BEAST_data.state > BEAST_burnin].to_numpy()
log_priors["BEAST"] = BEAST_data.prior[BEAST_data.state > BEAST_burnin].to_numpy()


# # Get data from VBPI

# In[21]:


for bs in ["10","20"]:
    self = VBPI_models[bs]
    n_runs = 1000
    n_particles = 1

    root_heights["VBPI_%s"%bs] = []
    tree_lengths["VBPI_%s"%bs] = []
    log_priors["VBPI_%s"%bs] = []
    log_likes["VBPI_%s"%bs] = []
    p_qs["VBPI_%s"%bs] = []

    for i in range(n_runs):
        with torch.no_grad():
            samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
            [namenum(tree, self.taxa) for tree in samp_trees]
            logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])

            samp_branch, logq_height, height, event_info = self.branch_model(samp_trees)
            log_clock_rate, logq_clock_rate = self.clock_model.sample(n_particles=n_particles)
            samp_branch = samp_branch.to(torch.float32) * log_clock_rate.exp()
            logll = torch.stack([self.phylo_model.loglikelihood(branch, tree) for branch, tree in zip(*[samp_branch, samp_trees])])

            self.tree_prior_model.update_batch(height, event_info)
            coalescent_param, logq_prior = self.tree_prior_model.sample_pop_size(n_particles=n_particles)
            logp_coalescent_prior, _ = self.tree_prior_model(coalescent_param, False)

            logp_clock_rate = self.clock_model(log_clock_rate)

            # get values
            root_heights["VBPI_%s"%bs].extend(list(height[:,0].numpy()))
            tree_lengths["VBPI_%s"%bs].extend(list(np.sum(samp_branch.numpy(),axis=1)))
            log_priors["VBPI_%s"%bs].extend(list(logp_coalescent_prior.numpy() + logp_clock_rate))
            log_likes["VBPI_%s"%bs].extend(list(logll.numpy()))
            p_qs["VBPI_%s"%bs].append((logll + logp_coalescent_prior + logp_clock_rate - logq_tree - logq_height - logq_prior - logq_clock_rate - math.log(n_particles)).item())


# # Get data from my code

with open(data_file, 'rb') as f:
    DS = pickle.load(f)

genomes = []
species = []
for key in DS:
    genomes.append(DS[key])
    species.append(key)

n_species = len(species)

# From https://github.com/zcrabbit/vbpi-torch/blob/ff86cf0c47a5753f5cc5b4dfe0b6ed783ab22669/unrooted/phyloModel.py#L7-L11
nuc2vec = {'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
           '-':[1.,1.,1.,1.], '?':[1.,1.,1.,1.], 'N':[1.,1.,1.,1.], 'R':[1.,1.,0.,0.],
           'Y':[0.,0.,1.,1.], 'S':[0.,1.,1.,0.], 'W':[1.,0.,0.,1.], 'K':[0.,1.,0.,1.],
           'M':[1.,0.,1.,0.], 'B':[0.,1.,1.,1.], 'D':[1.,1.,0.,1.], 'H':[1.,0.,1.,1.],
           'V':[1.,1.,1.,0.], '.':[1.,1.,1.,1.], 'U':[0.,0.,0.,1.], 'n':[1.,1.,1.,1.]}

for key in list(nuc2vec):
    nuc2vec[key.lower()] = nuc2vec[key]

tree_log_probs = torch.tensor([[nuc2vec[g] for g in genome] for genome in genomes],
                                dtype = torch.float64)
tree_log_probs = torch.log(tree_log_probs)

# In[22]:


def get_tree_length(tree):
    tree_length = 0
    for leaf in tree.leaves:
        tree_length += leaf.parent.coal_time.item() - leaf.coal_time

    for node in tree.nodes[:-1]:
        tree_length += node.parent.coal_time.item() - node.coal_time.item()

    return tree_length


# In[23]:


import gc
n_exp = 1000

for model in ["reinforce","reparam","VIMCO"]:
    print(model)
    with torch.no_grad():
        theta = optims[model].theta

        root_heights[model] = []
        tree_lengths[model] = []
        log_priors[model] = []
        log_likes[model] = []
        p_qs[model] = []

        for i in range(n_exp):

            Z = torch.normal(mean=0.0,std=1.0,size=(n_species,n_species))
            log_times = torch.exp(theta[1])*Z+theta[0]
            log_times = log_times + torch.triu(torch.full((n_species,n_species), float("Inf")))

            tree_log_probs0 = deepcopy(tree_log_probs)
            tree = Tree(theta,log_times,tree_log_probs0,
                        pop_size=pop_size)

            p_qs[model].append(tree.log_p.item() - tree.log_q.item())
            log_likes[model].append(tree.log_like.item())
            log_priors[model].append(tree.log_prior.item())
            root_heights[model].append(tree.nodes[-1].coal_time.item())
            tree_lengths[model].append(get_tree_length(tree))

            del tree

        gc.collect()


# # Recreate Figure 9 from Zhang et al (2024)

# In[24]:

# get bootstrap standard error of our models
def ELBO_se(p_qs):

    ELBO_hats = []

    for _ in range(1000):
        boot_sample = np.random.choice(p_qs,size=len(p_qs),replace=True)
        boot_ELBO = np.mean(boot_sample)
        ELBO_hats.append(boot_ELBO)

    return np.std(ELBO_hats,ddof=1)

def MLL_se(p_qs):

    MLL_hats = []

    for _ in range(1000):
        boot_sample = np.random.choice(p_qs,size=len(p_qs),replace=True)
        boot_MLL = logsumexp(boot_sample) - np.log(len(boot_sample))
        MLL_hats.append(boot_MLL)

    return np.std(MLL_hats,ddof=1)


# In[26]:


data = {'model': ["BEAST"] + [key for key in p_qs],
        'MLL': [np.mean(BEAST_MLLs)] + [logsumexp(p_qs[key]) - np.log(len(p_qs[key])) for key in p_qs],
        'MLL_se': [np.std(BEAST_MLLs,ddof=1)] + [MLL_se(p_qs[key]) for key in p_qs],
        'ELBO': [np.nan] + [np.mean(p_qs[key]) for key in p_qs],
        'ELBO_se': [np.nan] + [ELBO_se(p_qs[key]) for key in p_qs]}
df = pd.DataFrame(data)

print("MLLs:")
print("")
for i,key in enumerate(p_qs):
    print("%s: %s"%(key,data['MLL'][i+1]))

print("")
print("ELBO:")
print("")
for i,key in enumerate(p_qs):
    print("%s: %s"%(key,data['ELBO'][i+1]))

# Save the DataFrame to a CSV file
df.to_csv('../plt/'+data_set+'/'+data_set+'_MLLs.csv', index=False)


# In[27]:

# # make pickles


# In[29]:


iters = {model: optims[model].epochs for model in ["reinforce","reparam","VIMCO"]}
iters["VBPI_10"] = VBPI_iters["10"]
iters["VBPI_20"] = VBPI_iters["20"]
with open('../results/'+data_set+'/iters.pickle', 'wb') as handle:
    pickle.dump(iters, handle, protocol=pickle.HIGHEST_PROTOCOL)

runtimes = {model: optims[model].run_times for model in ["reinforce","reparam","VIMCO"]}
runtimes["VBPI_10"] = VBPI_runtimes["10"]
runtimes["VBPI_20"] = VBPI_runtimes["20"]
with open('../results/'+data_set+'/runtimes.pickle', 'wb') as handle:
    pickle.dump(runtimes, handle, protocol=pickle.HIGHEST_PROTOCOL)

MLLs = {model: [x.item() for x in optims[model].multi_ELBO_ests] for model in ["reinforce","reparam","VIMCO"]}
MLLs["VBPI_10"] = VBPI_lbs["10"]
MLLs["VBPI_20"] = VBPI_lbs["20"]

with open('../results/'+data_set+'/MLLs.pickle', 'wb') as handle:
    pickle.dump(MLLs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../results/'+data_set+'/BEAST_MLLs.pickle', 'wb') as handle:
    pickle.dump(BEAST_MLLs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../results/'+data_set+'/tree_lengths.pickle', 'wb') as handle:
    pickle.dump(tree_lengths, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../results/'+data_set+'/root_heights.pickle', 'wb') as handle:
    pickle.dump(root_heights, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../results/'+data_set+'/log_likes.pickle', 'wb') as handle:
    pickle.dump(log_likes, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../results/'+data_set+'/log_priors.pickle', 'wb') as handle:
    pickle.dump(log_priors, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../results/'+data_set+'/log_p_minus_log_q.pickle', 'wb') as handle:
    pickle.dump(p_qs, handle, protocol=pickle.HIGHEST_PROTOCOL)

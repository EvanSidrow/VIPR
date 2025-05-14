import numpy as np
import torch
import argparse
import pickle
import datetime
import os

from pathlib import Path
from io import StringIO
from Bio import Phylo

from VIPR import VIPR

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# extract parameters from command line
parser.add_argument('--prev_file', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--method', type=str, default=None)
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--rand_seed', type=int, default=None)
parser.add_argument('--pop_size', type=float, default=5.0)
parser.add_argument('--max_time', type=float, default=12.0)
parser.add_argument('--max_iters', type=int, default=10000)

args = parser.parse_args()

dataset = args.dataset
method = args.method
alpha = args.alpha
rand_seed = args.rand_seed
pop_size = args.pop_size
max_iters = args.max_iters
max_time = args.max_time

np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

# model parameters
var_dist = "LogNormal" #["LogNormal","Exponential","Mixture"]
rate = 1.0 # rate of evolution

# optimization parameters
batch_size = 10
record_every = 100
test_batch_size = 100

# decay rate parameters
decay = "exp" 
linear_decay = False
lr_decay_freq = 1
lr_decay_rate = 0.01**(1.0/max_iters) 

# select output file
time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
data_file = 'dat/'+dataset+'/'+dataset+'.pickle'
out_file = 'results/'+dataset+'/'+dataset+'_'+var_dist+'_'+method+'_'+str(alpha)+'_'+str(rand_seed)+'_'+time+'.pickle'
Path('results/'+dataset).mkdir(parents=True, exist_ok=True)
if os.path.isfile(out_file):
    print("file %s already exits. Exiting..."%out_file)
    exit()

# print off initial values
print("dataset: ", dataset)
print("method: ", method)
print("initial step size: ", alpha)
print("random seed: ", rand_seed)
print("output file: ", out_file)
print("")

### Import Data ###

with open(data_file, 'rb') as f:
    ds = pickle.load(f)

genomes = []
species = []
for key in ds:
    genomes.append(ds[key])
    species.append(key)

ntaxa = len(species)

### Initialize Phi ###

print("Initializing phi... \n")
treedata = ""
ntrees = 0
burnin = 10000

for i in range(10):
    tree_file = "dat/"+dataset+"/"+dataset+"_fixed_pop_support_short_run_rep_%d.trees"%(i+1)
    with open(tree_file, "r") as file:
        for j,line in enumerate(file):
            if line.startswith("tree STATE") and j%10 == 0 and int(line.split("_")[1].split()[0]) > burnin:
                line = line[line.find('('):]
                line = line.replace("[&rate=1.0]","")
                line = line.replace("[&rate=0.001]","")
                treedata = treedata + line + "\n"
                ntrees += 1

phi0 = torch.zeros((2,ntaxa,ntaxa))
trees = Phylo.parse(StringIO(treedata), "newick")
dists = np.zeros((ntrees,ntaxa,ntaxa))

for i,tree in enumerate(trees):
    for j in range(ntaxa):
        for k in range(j):
            mrca = tree.common_ancestor(str(j+1),str(k+1))
            dists[i,j,k] = min(tree.distance(mrca,str(j+1)),tree.distance(mrca,str(k+1)))

for j in range(ntaxa):
    for k in range(j):
        phi0[0,j,k] = np.mean(dists[:,j,k])
        phi0[1,j,k] = np.var(dists[:,j,k])

if rand_seed > 0:
    phi0 = phi0 + torch.normal(mean=0.0,std=rand_seed*0.1,size=(2,ntaxa,ntaxa))

### Train Model ###

if args.prev_file:
    with open(args.prev_file, 'rb') as f:
        optim = pickle.load(f)
else:
    optim = VIPR(genomes,phi0[0],phi0[1],var_dist=var_dist,
                phi_pop_size=torch.tensor([pop_size]),var_dist_pop_size="Fixed",
                theta_pop_size=None,prior_pop_size="Fixed",
                tip_dates=None,
                phi_rate=torch.tensor([rate]),var_dist_rate="Fixed",
                theta_rate=None,prior_rate="Fixed")

print("Training model... \n")
optim.learn(batch_size=batch_size,
            iters=max_iters,
            alpha=alpha,
            method=method,
            record_every=record_every,
            test_batch_size=test_batch_size,
            pop_size=pop_size,
            lr_decay_freq=lr_decay_freq,
            lr_decay_rate=lr_decay_rate,
            linear_decay=linear_decay,
            max_time=max_time)

with open(out_file, 'wb') as file:
    pickle.dump(optim, file)

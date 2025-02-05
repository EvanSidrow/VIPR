import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from itertools import product
from itertools import combinations
import pickle
import datetime
from scipy.special import logsumexp
import os

from Bio.Phylo.TreeConstruction import _DistanceMatrix
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceCalculator
from io import StringIO
from Bio import Phylo

from tree_torch import Tree
from SLCVI_torch import SLCVI

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# turn the process id into a set of parameters
parser.add_argument('--pid', type=int, required=True, help=' HCV ')
parser.add_argument('--prev_file', type=str, default=None, help=' HCV ')
args = parser.parse_args()

datasets = ["DS1","DS2","DS3","DS4","DS5","DS6",
            "DS7","DS8","DS9","DS10","DS11","DS14"]
#datasets = ["DS14"]
methods = ["reparam","reinforce","VIMCO"]
alphas = [0.03,0.01,0.003,0.001]

method = methods[args.pid % 3]
alpha = alphas[int(args.pid/3) % 4]
dataset = datasets[int(args.pid/12) % 12]
rand_seed = int(args.pid/144)

np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

# keep fixed values
decay = "exp"
batch_size = 10
max_iters = 10000
record_every = 100
test_batch_size = 500
if decay == "linear":
    linear_decay = True
else:
    linear_decay = False
anneal_freq = 1
anneal_rate = 0.01**(1.0/10000)
pop_size = 5.0
max_time = 24.0 # HOURS

# select output file
time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
data_file = 'dat/'+dataset+'/'+dataset+'.pickle'
out_file = 'results/'+dataset+'/'+dataset+'_'+method+'_'+str(alpha)+'_'+str(rand_seed)+'_'+time+'.pickle'

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

with open(data_file, 'rb') as f:
    ds = pickle.load(f)

genomes = []
species = []
for key in ds:
    genomes.append(ds[key])
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

#######

treedata = ""
ntrees = 0

for i in range(10):
    tree_file = "dat/"+dataset+"/"+dataset+"_fixed_pop_support_short_run_rep_%d.trees"%(i+1)
    with open(tree_file, "r") as file:
        for j,line in enumerate(file):
            if j%10 == 0 and line.startswith("tree STATE"):

                # start at first (
                line = line[line.find('('):]
                # remove rate infor
                line = line.replace("[&rate=1.0]","")
                treedata = treedata + line + "\n"
                ntrees += 1

theta = torch.zeros((2,n_species,n_species))
trees = Phylo.parse(StringIO(treedata), "newick")
log_dists = np.zeros((ntrees,n_species,n_species))

for i,tree in enumerate(trees):
    for j in range(n_species):
        for k in range(j):
            log_dists[i,j,k] = np.log(tree.distance(target1=str(j+1),target2=str(k+1))/2.0)

for j in range(n_species):
    for k in range(j):
        theta[0,j,k] = np.mean(log_dists[:,j,k])
        theta[1,j,k] = np.log(np.std(log_dists[:,j,k]))

#theta = torch.zeros((2,n_species,n_species))
#times = torch.zeros((n_species,n_species))
#for i in range(n_species):
#    for j in range(n_species):
#        eq = [x != y for x,y in zip(genomes[i],genomes[j]) if ((x in ["A","C","T","G"]) and (y in ["A","C","T","G"]))]
#        times[i,j] = np.mean(eq) + 1/(len(eq))
#log_times = torch.log(times) + torch.triu(torch.full((n_species,n_species), float("Inf")))
#tree = Tree(theta,log_times,tree_log_probs,
#            pop_size=pop_size)

#for node in tree.nodes:
#    combs = combinations(node.leaves,2)
#    for inds in combs:
#        inds = sorted(inds)
#        theta[0,inds[1],inds[0]] = min(theta[0,inds[1],inds[0]],torch.log(node.coal_time))
#        theta[1,inds[1],inds[0]] = -2

# add random noise
if rand_seed > 0:
    theta = theta + torch.normal(mean=0.0,std=rand_seed*0.1,size=(2,n_species,n_species))

#######

if args.prev_file:
    with open(args.prev_file, 'rb') as f:
        optim = pickle.load(f)
else:
    optim = SLCVI(tree_log_probs,theta,pop_size)

optim.learn(batch_size=batch_size,
            iters=max_iters,
            alpha=alpha,
            method=method,
            record_every=record_every,
            test_batch_size=test_batch_size,
            pop_size=pop_size,
            anneal_freq=anneal_freq,
            anneal_rate=anneal_rate,
            linear_decay=linear_decay,
            max_time=max_time)

with open(out_file, 'wb') as file:
    pickle.dump(optim, file)

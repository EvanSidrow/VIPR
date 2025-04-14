import numpy as np
import torch
import argparse
import pickle
import datetime
import os

from io import StringIO
from Bio import Phylo

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VIPR import VIPR

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# turn the process id into a set of parameters
parser.add_argument('--pid', type=int, required=True)
parser.add_argument('--prev_file', type=str, default=None)
args = parser.parse_args()

datasets = ["taxa_00008","taxa_00016","taxa_00032",
            "taxa_00064","taxa_00128","taxa_00256",
            "taxa_00512"]
#datasets = ["DS14_3","DS14_6","DS14_12","DS14_24","DS14_48","DS14_72"]
#datasets = ["taxa_00512","taxa_01024"]
methods = ["reparam","reinforce","VIMCO"]

method = methods[args.pid % 3]
dataset = datasets[int(args.pid/3) % 7]
rand_seed = 0
alpha = 0.03

np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

# keep fixed values
decay = "exp"
batch_size = 10
max_iters = 1000
record_every = 100
test_batch_size = 100
if decay == "linear":
    linear_decay = True
else:
    linear_decay = False
anneal_freq = 1
anneal_rate = 0.01**(1.0/max_iters)
pop_size = 5.0
max_time = 5.0/60.0 # HOURS

# select output file
time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
data_file = 'dat/'+dataset+'/'+dataset+'.pickle'
out_file = 'results/'+dataset+'/'+dataset+'_'+method+'_'+str(alpha)+'_'+str(rand_seed)+'_'+time+'.pickle'

if not os.path.exists('results/'+dataset):
    os.makedirs('results/'+dataset)

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

ntaxa = len(species)

#######

treedata = ""
ntrees = 0

print("Initializing theta... \n")
for i in range(1):
    tree_file = "dat/"+dataset+"/"+dataset+"_fixed_pop_support_short_run_rep_%d.trees"%(i+1)
    with open(tree_file, "r") as file:
        for j,line in enumerate(file):
            if j%1000 == 999 and line.startswith("tree STATE"):
                line = line[line.find('('):]
                line = line.replace("[&rate=1.0]","")
                treedata = treedata + line + "\n"
                ntrees += 1

theta = torch.zeros((2,ntaxa,ntaxa))
trees = Phylo.parse(StringIO(treedata), "newick")
dists = np.zeros((ntrees,ntaxa,ntaxa))

print("number of trees: ", ntrees)

for i,tree in enumerate(trees):
    print("tree %d of %d"%(i+1,ntrees))
    for j in range(ntaxa):
        for k in range(j):
            dists[i,j,k] = tree.distance(target1=str(j+1),target2=str(k+1))/2.0

for j in range(ntaxa):
    for k in range(j):
        theta[0,j,k] = np.mean(dists[:,j,k])
        theta[1,j,k] = np.exp(-4)#np.var(dists[:,j,k])

# add random noise
if rand_seed > 0:
    theta = theta + torch.normal(mean=0.0,std=rand_seed*0.1,size=(2,ntaxa,ntaxa))

#######

if args.prev_file:
    with open(args.prev_file, 'rb') as f:
        optim = pickle.load(f)
else:
    optim = VIPR(genomes,theta[0],theta[1],pop_size)

print("Training model.. \n")
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
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

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()

# turn the process id into a set of parameters
parser.add_argument('--pid', type=int, required=True, help=' HCV ')
args = parser.parse_args()

datasets = ["DS1","DS2","DS3","DS4","DS5","DS6",
            "DS7","DS8","DS9","DS10","DS11"]
methods = ["reparam","reinforce","reinforce_VIMCO"]
alphas = [0.1,0.03,0.01,0.003,0.001,0.0003,0.0001]
decays = ["none","linear","exp"]

method = methods[args.pid % 3]
alpha = alphas[int(args.pid/3) % 7]
decay = decays[int(args.pid/21) % 3]
dataset = datasets[int(args.pid/63) % 11]

# keep fixed values
batch_size = 10
max_iters = int(100000 / batch_size)
record_every = 10
test_batch_size = 5*batch_size
if decay == "linear":
    linear_decay = True
else:
    linear_decay = False
anneal_freq = 1
if decay == "exp":
    anneal_rate = 0.01**(1.0/max_iters)
else:
    anneal_rate = 1.0
pop_size = 5.0
max_time = 12.0 # HOURS

# select output file
time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
data_file = 'dat/'+dataset+'/'+dataset+'.pickle'
out_file = 'results/'+dataset+'/'+dataset+'_'+method+'_'+decay+'_'+str(alpha)+'_'+time+'.pickle'

if os.path.isfile(out_file):
    print("file %s already exits. Exiting..."%out_file)
    exit()

# print off initial values
print("dataset: ", dataset)
print("method: ", method)
print("batch size: ", batch_size)
print("initial step size: ", alpha)
print("decay: ", decay)
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

tree_log_probs = torch.tensor([[nuc2vec[g] for g in genome] for genome in genomes],
                                dtype = torch.float64)
tree_log_probs = torch.log(tree_log_probs)

#######

times = np.zeros((n_species,n_species))

for i in range(n_species):
    for j in range(n_species):
        eq = [x != y for x,y in zip(genomes[i],genomes[j]) if ((x in ["A","C","T","G"]) and (y in ["A","C","T","G"]))]
        p_hat = np.mean(eq)

        times[i,j] = np.mean(eq)

matrix = []
for i,row in enumerate(times):
    matrix.append(list(row[:(i+1)]))
m = _DistanceMatrix(species, matrix)
calculator = DistanceCalculator('identity')
constructor = DistanceTreeConstructor(calculator, 'nj')
tree = constructor.nj(m)

theta = torch.zeros((2,n_species,n_species))
for i in range(n_species):
    for j in range(i):
        n = sum([((x in ["A","C","T","G"]) and (y in ["A","C","T","G"])) for x,y in zip(genomes[i],genomes[j])])
        m = tree.distance(target1=species[i],target2=species[j]) + (1/(n+2))
        v = m*(1.0-m) / (n+1)
        sig2 = np.log(v/(m*m) + 1.0)
        mu = np.log(m) - sig2/2.0
        theta[0,i,j] = mu
        theta[1,i,j] = np.log(sig2)/2.0

#######

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

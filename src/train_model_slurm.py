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
batch_sizes = [10,20,100]
methods = ["reparam","reinforce","VIMCO"]
alphas = [0.2,0.02,0.002,0.0002]

dataset = datasets[args.pid % 11]
batch_size = batch_sizes[int(args.pid/11) % 3]
method = methods[int(args.pid/33) % 3]
alpha = alphas[int(args.pid/99) % 4]

# keep fixed values
max_iters = int(100000 / batch_size)
record_every = 10
test_batch_size = 5*batch_size
linear_decay = True
anneal_freq = 1
anneal_rate = 1.0
pop_size = 5.0

# select output file
time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
data_file = 'dat/'+dataset+'/'+dataset+'.pickle'
out_file = 'results/'+dataset+'/'+dataset+'_'+method+'_'+str(batch_size)+'_'+str(alpha)+'.pickle'

# print off initial values
print("dataset: ", dataset)
print("method: ", method)
print("batch size: ", batch_size)
print("initial step size: ", alpha)
print("output file: ", out_file)
print("")

if not os.path.exists('../results/'+dataset):
    os.makedirs('../results/'+dataset)

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
           'V':[1.,1.,1.,0.], '.':[1.,1.,1.,1.], 'U':[0.,0.,0.,1.]}

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
        theta[0,i,j] = np.log(tree.distance(target1=species[i],target2=species[j]))
        theta[1,i,j] = -2
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
            linear_decay=linear_decay)

with open(out_file, 'wb') as file:
    pickle.dump(optim, file)

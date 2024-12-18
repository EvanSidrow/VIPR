import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from itertools import product
from itertools import combinations
import pickle
import datetime
from scipy.special import logsumexp

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


########## Data arguments
parser.add_argument('--dataset', required=True, help=' HCV ')
parser.add_argument('--batch_size', type=int, required=True, help=' HCV ')
parser.add_argument('--max_iters', type=int, required=True, help=' HCV ')
parser.add_argument('--record_every', type=int, required=True)
parser.add_argument('--test_batch_size', type=int, required=True)
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--anneal_freq', type=int, default=1, help=' HCV ')
parser.add_argument('--anneal_rate', type=float, default=1.0, help=' HCV ')
parser.add_argument('--linear_decay', type=bool, default=False, help=' HCV ')
parser.add_argument('--alpha', type=float, default=0.1, help=' HCV ')
parser.add_argument('--pop_size', type=float, default=5.0, help=' HCV ')

args = parser.parse_args()

data_set = args.dataset
batch_size = args.batch_size
max_iters = args.max_iters
pop_size = args.pop_size

time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
data_file = 'dat/'+data_set+'/'+data_set+'.pickle'
out_file = 'results/'+data_set+'/'+data_set+'_'+time+'_'+args.method+'.pickle'

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
        theta[0,i,j] = np.log(tree.distance(target1=species[i],target2=species[j]))
        theta[1,i,j] = -2
#######

optim = SLCVI(tree_log_probs,theta,pop_size)
optim.learn(batch_size=args.batch_size,
            iters=args.max_iters,
            alpha=args.alpha,
            method=args.method,
            record_every=args.record_every,
            test_batch_size=args.test_batch_size,
            pop_size=args.pop_size,
            anneal_freq=args.anneal_freq,
            anneal_rate=args.anneal_rate,
            linear_decay=args.linear_decay)

with open(out_file, 'wb') as file:
    pickle.dump(optim, file)

import autograd.numpy as np
from autograd.scipy.special import logsumexp,erf
from autograd.scipy.stats import norm
from autograd import grad

import argparse
#from autograd_gamma import gamma, gammainc, gammaincc, gammaincln, gammainccln

import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import product
from itertools import combinations
import pickle
import datetime

from Bio.Phylo.TreeConstruction import _DistanceMatrix
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceCalculator
from io import StringIO
from Bio import Phylo

from tree import Tree
from optimizer import Optimizer

import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)

parser = argparse.ArgumentParser()


########## Data arguments
parser.add_argument('--dataset', required=True, help=' HCV ')
parser.add_argument('--batch_size', type=int, required=True, help=' HCV ')
parser.add_argument('--max_iters', type=int, required=True, help=' HCV ')
parser.add_argument('--alpha_start', type=float, default=0.1, help=' HCV ')
parser.add_argument('--alpha_end', type=float, default=0.01, help=' HCV ')
parser.add_argument('--pop_size', type=float, default=5.0, help=' HCV ')

args = parser.parse_args()

data_set = args.dataset
batch_size = args.batch_size
max_iters = args.max_iters
pop_size = args.pop_size

time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
data_file = 'dat/'+data_set+'/'+data_set+'.pickle'
reparam_file = 'results/'+data_set+'/'+data_set+'_'+time+'_reparam.pickle'
reinforce_file = 'results/'+data_set+'/'+data_set+'_'+time+'_reinforce.pickle'
VIMCO_file = 'results/'+data_set+'/'+data_set+'_'+time+'_VIMCO.pickle'

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

tree_log_probs = np.array([[nuc2vec[g] for g in genome] for genome in genomes],
                                dtype = float)
tree_log_probs = np.log(tree_log_probs)

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

theta = np.zeros((2,n_species,n_species))
for i in range(n_species):
    for j in range(i):
        theta[0,i,j] = np.log(tree.distance(target1=species[i],target2=species[j]))
        theta[1,i,j] = -2
#######

optim_reparam = Optimizer(tree_log_probs,deepcopy(theta))
optim_reparam.optimize_q_reparam(batch_size=batch_size,
                                 iters=max_iters,
                                 alphas=[args.alpha_start,args.alpha_end],
                                 record_every=10,
                                 test_batch_size=5*batch_size,
                                 pop_size=pop_size)

with open(reparam_file, 'wb') as file:
    pickle.dump(optim_reparam, file)


optim_reinforce = Optimizer(tree_log_probs,deepcopy(optim_reparam.theta))
optim_reinforce.optimize_q_reinforce(batch_size=batch_size,
                                     iters=max_iters,
                                     alphas=[args.alpha_start,args.alpha_end],
                                     record_every=10,
                                     test_batch_size=5*batch_size,
                                     pop_size=pop_size)

with open(reinforce_file, 'wb') as file:
    pickle.dump(optim_reinforce, file)


optim_VIMCO = Optimizer(tree_log_probs,deepcopy(optim_reparam.theta))
optim_VIMCO.optimize_q_VIMCO(batch_size=batch_size,
                             iters=max_iters,
                             alphas=[args.alpha_start,args.alpha_end],
                             record_every=10,
                             test_batch_size=5*batch_size,
                             pop_size=pop_size)

with open(VIMCO_file, 'wb') as file:
    pickle.dump(optim_VIMCO, file)

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
parser.add_argument('--alpha_reparam', type=float, default=0.1, help=' HCV ')
parser.add_argument('--alpha_reinforce', type=float, default=0.01, help=' HCV ')
parser.add_argument('--pop_size', type=float, default=5.0, help=' HCV ')

args = parser.parse_args()

data_set = args.dataset
batch_size = args.batch_size
max_iters = args.max_iters
pop_size = args.pop_size

data_file = 'dat/'+data_set+'/'+data_set+'.pickle'
reparam_file = 'results/'+data_set+'/'+data_set+'_reparam.pickle'
reinforce_file = 'results/'+data_set+'/'+data_set+'_reinforce.pickle'

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

tree_log_probs = np.array([[nuc2vec[g] for g in genome] for genome in genomes],
                                dtype = float)
tree_log_probs = np.log(tree_log_probs)

theta = np.zeros((2,n_species,n_species))

for i in range(n_species):
    for j in range(i):
        eq = [x != y for x,y in zip(genomes[i],genomes[j]) if ((x in ["A","C","T","G"]) and (y in ["A","C","T","G"]))]
        p_hat = np.mean(eq)

        theta[0,i,j] = np.log(p_hat) - np.log(2)
        theta[1,i,j] = -1

optim_reparam = Optimizer(tree_log_probs,deepcopy(theta))
optim_reparam.optimize_q_reparam(batch_size=batch_size,
                                 iters=max_iters,
                                 alphas=[args.alpha_reparam,args.alpha_reparam],
                                 record_every=np.floor(max_iters/50),
                                 test_batch_size=50,
                                 pop_size=pop_size)

with open(reparam_file, 'wb') as file:
    pickle.dump(optim_reparam, file)


optim_reinforce = Optimizer(tree_log_probs,deepcopy(optim_reparam.theta))
optim_reinforce.optimize_q_reinforce(batch_size=batch_size,
                                     iters=max_iters,
                                     alphas=[args.alpha_reinforce,args.alpha_reinforce],
                                     record_every=50,
                                     test_batch_size=5*batch_size,
                                     pop_size=pop_size)

with open(reinforce_file, 'wb') as file:
    pickle.dump(optim_reinforce, file)

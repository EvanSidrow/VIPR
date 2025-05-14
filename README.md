# VIPR: Variational Phylogenetic Inference with Products over Bipartitions

This repository contains code required to run experiments in the paper "Variational Phylogenetic Inference with Products over Bipartitions" by Evan Sidrow, Alexandre Bouchard-Côté, and Lloyd T. Elliott.

All of the experiments below can (and should) be run in parallel and from the root directory. Requirements are detailed in the requirements.txt file in the root directory. 

Python version is 3.13.
Beast version is 1.10.4.

All experiments are set to run for 10,000 iterations, but it is straightforward to increase the number of iterations by changing the command line arguments below.

## Running primary experiments

### BEAST

Run the following command in a terminal:

`bash src/run_beast.sh DATASET` 

for all datasets: DS1, DS2, DS3, DS4, DS5, DS6, DS7, DS8, DS9, DS10, DS11, COV 

### VBPI

Run the following command in a terminal:

`python src/vbpi-torch/rooted/main.py --dataset DATASET --alpha ALHPA --nParticle NPARTICLE --rand_seed RAND_SEED --maxIter 10000 --coalescent_type fixed_pop --clock_type fixed_rate --init_clock_rate 1.0 --log_pop_size_offset 1.6094379124341003 --burnin 250 --psp`

with all combninations of the following settings:

    - datasets: DS1, DS2, DS3, DS4, DS5, DS6, DS7, DS8, DS9, DS10, DS11, COV
    - alphas: 0.003, 0.001, 0.0003, 0.0001
    - nParticles: 10, 20
    - random seeds: 0-9

### VIPR

Run the following command in a terminal: 

`python src/train_model.py --dataset DATASET --method METHOD --alpha ALPHA --rand_seed RAND_SEED --max_iters 10000`

with all combinations of the following settings:

    - datasets: DS1, DS2, DS3, DS4, DS5, DS6, DS7, DS8, DS9, DS10, DS11, COV
    - methods: reparam, reinforce, VIMCO
    - alphas: 0.03, 0.01, 0.003, 0.001
    - random seeds: 0-9

### Plotting results

Run the "plot_results.ipynb" jupyter notebook to interactively plot the results.


## Running computational complexity experiments:

### BEAST

`bash src/run_beast_comp.sh DATASET` 

for all datasets: taxa_00008, taxa_00016, taxa_00032, taxa_00064, taxa_00128, taxa_00256, taxa_00512

### VBPI

Run the following command in a terminal:

`python src/vbpi-torch/rooted/main.py --dataset DATASET --nParticle NPARTICLE --alpha 0.0001 --rand_seed 0 --maxIter 1000 --max_time 0.1 --coalescent_type fixed_pop --clock_type fixed_rate --init_clock_rate 1.0 --log_pop_size_offset 1.6094379124341003 --burnin 250 --psp`

with all combninations of the following settings:

    - datasets: taxa_00008, taxa_00016, taxa_00032, taxa_00064, taxa_00128, taxa_00256, taxa_00512
    - nParticles: 10, 20

### VIPR

Run the following in a terminal:

`python src/train_model.py --dataset DATASET --method METHOD --alpha 0.001 --rand_seed 0 --max_time 0.1 --max_iters 1000`

with all combinations of the following settings:

    - datasets: taxa_00008, taxa_00016, taxa_00032, taxa_00064, taxa_00128, taxa_00256, taxa_00512
    - methods: reparam, reinforce, VIMCO

### plot results

Run the "plot_comp.ipynb" jupyter notebook to interactively plot the results.

## cubeVB coverage experiments:

Run the "cubeVB_coverage.ipynb" jupyter notebook to interactively plot the results.
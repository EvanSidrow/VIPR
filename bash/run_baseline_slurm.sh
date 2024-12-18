#!/bin/bash
#SBATCH --account=def-nheckman

#SBATCH --mail-user=evan.sidrow@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-131

module load python/3.9
module load scipy-stack

python src/vbpi-torch/rooted/main_slurm.py --pid $SLURM_ARRAY_TASK_ID \
--coalescent_type fixed_pop --clock_type fixed_rate --init_clock_rate 1.0 \
--log_pop_size_offset 1.6094379124341003 --burnin 250 --psp

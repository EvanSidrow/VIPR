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
#SBATCH --array=0-719

module load StdEnv/2023
module load scipy-stack/2024b

# tell the cpu and model name on linux
cat /proc/cpuinfo | grep "model name" | head -n1 | cut -d ":" -f2 | cut -d " " -f2

python src/train_model.py --pid $SLURM_ARRAY_TASK_ID

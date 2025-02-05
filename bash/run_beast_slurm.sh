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
#SBATCH --array=1-13

# add correct modules
module load StdEnv/2023
module load java/1.8.0_292
module load cuda

# change to correct folder
cd dat/DS${SLURM_ARRAY_TASK_ID}

# run stepping stones
for i in $(seq 1 10);
do
  java -jar ../../BEASTv1.10.4/lib/beast.jar -overwrite -prefix "${i}_MLL_" -seed $i "DS${SLURM_ARRAY_TASK_ID}_MLL.xml" > "DS${SLURM_ARRAY_TASK_ID}_MLL_${i}.txt"
done

for i in $(seq 1 10);
do
  mv "${i}_MLL_DS${SLURM_ARRAY_TASK_ID}.trees" "DS${SLURM_ARRAY_TASK_ID}_fixed_pop_MLL_${i}.trees"
  mv "${i}_MLL_DS${SLURM_ARRAY_TASK_ID}.log" "DS${SLURM_ARRAY_TASK_ID}_fixed_pop_MLL_${i}.log"
  mv "${i}_MLL_DS${SLURM_ARRAY_TASK_ID}.ops" "DS${SLURM_ARRAY_TASK_ID}_fixed_pop_MLL_${i}.ops"
done

# run psp
for i in $(seq 1 10);
do
  java -jar ../../BEASTv1.10.4/lib/beast.jar -overwrite -prefix "${i}_" -seed $i "DS${SLURM_ARRAY_TASK_ID}.xml"
done

for i in $(seq 1 10);
do
  mv "${i}_DS${SLURM_ARRAY_TASK_ID}.trees" "DS${SLURM_ARRAY_TASK_ID}_fixed_pop_support_short_run_rep_${i}.trees"
  mv "${i}_DS${SLURM_ARRAY_TASK_ID}.log" "DS${SLURM_ARRAY_TASK_ID}_fixed_pop_support_short_run_rep_${i}.log"
  mv "${i}_$DS${SLURM_ARRAY_TASK_ID}.ops" "DS${SLURM_ARRAY_TASK_ID}_fixed_pop_support_short_run_rep_${i}.ops"
done

# run ground truth
java -jar ../../BEASTv1.10.4/lib/beast.jar -overwrite -prefix golden_ "$1_golden.xml"
mv "golden_DS${SLURM_ARRAY_TASK_ID}.trees" "DS${SLURM_ARRAY_TASK_ID}_fixed_pop_golden_run.trees"
mv "golden_DS${SLURM_ARRAY_TASK_ID}.log" "DS${SLURM_ARRAY_TASK_ID}_fixed_pop_golden_run.log"
mv "golden_DS${SLURM_ARRAY_TASK_ID}.ops" "DS${SLURM_ARRAY_TASK_ID}_fixed_pop_golden_run.ops"

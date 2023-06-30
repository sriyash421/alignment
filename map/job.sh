#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=5
#SBATCH --mem=31G
#SBATCH --time=23:50:00
#SBATCH --array=0-2


cd $HOME/alignment
source venv/bin/activate
module load opencv

task=$1
agents=$2

cd map
python train_multi_sacd.py --task $task --num-good-agents $agents --obs-radius 0.5 --intr-rew elign_team --epoch 100  --benchmark  --logdir log/$task_${SLURM_ARRAY_TASK_ID} --seed ${SLURM_ARRAY_TASK_ID} --wandb-enabled

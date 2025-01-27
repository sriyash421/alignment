#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=71:50:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH -o /home/sriyash/scratch/ELIGN_FINAL_LOGS/%j.out

cd $HOME/alignment
source venv/bin/activate
module load opencv

task=$1
agents=$2
int_rew=$3

cd map
python train_multi_sacd.py --task $task --num-good-agents $agents --obs-radius 0.5 --intr-rew $int_rew --epoch 100  --benchmark  --logdir $SCRATCH/ELIGN_FINAL/$task_$agents_$3_${SLURM_ARRAY_TASK_ID} --seed ${SLURM_ARRAY_TASK_ID} --wandb-enabled --save-models

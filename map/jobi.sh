#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=43:50:00
#SBATCH --array=0-4
#SBATCH -o /home/sriyash/elign_res/%j.out

cd $HOME/alignment
source venv/bin/activate
module load opencv

task=$1
agents=$2
int_rew=$3
beta=$4
temp=$5

cd map
echo "python train_multi_sacd_int.py --beta $beta --temp $temp  --task $task --num-good-agents $agents --obs-radius 0.5 --intr-rew $int_rew --epoch 100  --benchmark  --logdir log/$task_${SLURM_ARRAY_TASK_ID} --seed ${SLURM_ARRAY_TASK_ID} --wandb-enabled"

python train_multi_sacd_int.py --beta $beta --temp $temp  --task $task --num-good-agents $agents --obs-radius 0.5 --intr-rew $int_rew --epoch 100  --benchmark  --logdir log/$task_${SLURM_ARRAY_TASK_ID} --seed ${SLURM_ARRAY_TASK_ID} --wandb-enabled

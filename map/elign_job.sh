#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=23:50:00
#SBATCH -o ${SCRATCH}/elign/ELIGN%j
#SBATCH --job-name=ELIGN%j


cd $HOME/alignment
source venv/bin/activate
module load opencv

python train_multi_sacd.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew elign_team --epoch 100 --save-models --benchmark  --logdir log/simple_spread

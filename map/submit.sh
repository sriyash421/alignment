
sbatch job.sh simple_spread_in 3 elign_self
sbatch job.sh hetero_spread_in 4 elign_self

sbatch job.sh simple_spread_in 3 elign_team
sbatch job.sh hetero_spread_in 4 elign_team

sbatch jobi.sh simple_spread_in 3 ours 0.01 0.001
sbatch jobi.sh hetero_spread_in 4 ours 0.01 0.001
sbatch jobi.sh simple_spread_in 3 ours -0.01 0.001
sbatch jobi.sh hetero_spread_in 4 ours -0.01 0.001

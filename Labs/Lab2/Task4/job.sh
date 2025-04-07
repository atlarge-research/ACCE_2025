#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=TitanX

./grid_strike_vectorAdd_exercise
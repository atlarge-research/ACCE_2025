#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=TitanX

module load cuda12.3/toolkit/12.3

nvidia-smi

nvidia-smi -q

/cm/shared/apps/cuda12.3/toolkit/12.3/extras/demo_suite/deviceQuery

#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=TitanX

module load cuda12.3/toolkit

nvprof --metrics issue_slot_utilization ./vectorAdd
#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --gpus=8                  
#SBATCH -w soketlab-node004       
#SBATCH --ntasks-per-node=1
#SBATCH --partition=tts
#SBATCH --qos=cpu60
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/inference_agri_%j.out
#SBATCH --error=logs/inference_agri_%j.err

module load gcc
module load cuda/12.8
module load python/3.10
module load nccl

source .agrillm/bin/activate

python scripts/inference_serve.py
#!/bin/bash
#SBATCH --job-name=serve-agri
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH -w soketlab-node003
#SBATCH --ntasks-per-node=1
#SBATCH --partition=tts
#SBATCH --qos=cpu60
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/litgpt_serve_%j.out
#SBATCH --error=logs/litgpt_serve_%j.err

module load gcc
module load cuda/12.8
module load python/3.10
module load nccl

source .agrillm/bin/activate

echo "------------------------------------------------------"
echo "Job running on node: $(hostname)"
echo "SLURM GPUs: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "------------------------------------------------------"

MODEL_DIR="/projects/data/teams/tts_team/agri_training/test_16000_out_dir_8_1_three_combined_dataset_gemma/step-004800"

echo "Starting LitGPT server"
echo "Checkpoint dir: $MODEL_DIR"

litgpt serve \
  $MODEL_DIR \
  --openai_spec true \
  --devices 8 \
  --max_new_tokens 4096 \
  --accelerator cuda \
  --precision bf16 \
  --port 8000

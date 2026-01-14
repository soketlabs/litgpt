#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=tts
#SBATCH --qos=cpu60
#SBATCH --nodelist=soketlab-node038,soketlab-node039
#SBATCH --time=4-20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err



module load gcc
module load cuda
module load nccl
module load python/3.10




# export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# export NCCL_NVLS_ENABLE=0s
# export NVTE_DP_AMAX_REDUCE_INTERVAL=0
# export NVTE_ASYNC_AMAX_REDUCTION=1  
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_ALGO= "Tree"
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export NCCL_SOCKET_IFNAME=bond0

# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

srun litgpt finetune_lora  --config config_hub/finetune/gemma3-27b-it/lora_2700.yaml

 
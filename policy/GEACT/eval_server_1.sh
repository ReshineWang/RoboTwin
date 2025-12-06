#!/bin/bash

# ===== 基础参数 =====
policy_name=GEACT
task_name=beat_block_hammer
task_config=demo_clean
ckpt_setting=demo_clean
expert_data_num=50
seed=42
gpu_id=0
port=55559

policy_conda_env=genie_envisioner

echo -e "\033[33m[SERVER] Using GPU: ${gpu_id}\033[0m"

export CUDA_VISIBLE_DEVICES=${gpu_id}
export GEACT_MODE=double
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# ===== 激活 conda 环境 =====
echo -e "\033[32m[SERVER] Activating Conda environment: ${policy_conda_env}\033[0m"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

cd /data/dex/RoboTwin

echo -e "\033[32m[SERVER] Launching policy_model_server...\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python script/policy_model_server.py \
    --config /data/dex/RoboTwin/policy/GEACT/deploy_policy.yml \
    --port ${port} \
    --overrides \
    --policy_name ${policy_name} \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --config /data/dex/RoboTwin/policy/GEACT/configs/ltx_model/policy_model_lerobot.yaml \
    --weight /data/dex/Genie-Envisioner/GE_ACT_ROBOTWIN_FINETUNE/2025_11_20_17_35_39/step_8000/diffusion_pytorch_model.safetensors \
    --denoise_step 10 \
    --domain_name RoboTwin


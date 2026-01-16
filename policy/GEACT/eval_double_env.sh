#!/bin/bash

policy_name=GEACT
task_name=beat_block_hammer
task_config=demo_clean
ckpt_setting=demo_clean
expert_data_num=50
seed=42
gpu_id=1
policy_conda_env=genie_envisioner
sim_conda_env=RoboTwin
port=55551
execution_step=140

export CUDA_VISIBLE_DEVICES=${gpu_id}
export GEACT_MODE=double  # 控制双环境推理
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

yaml_file="policy/${policy_name}/deploy_policy.yml"

echo "policy_conda_env is '$policy_conda_env'"

# Start the server in the background
echo -e "\033[32m[server] Activating Conda environment: ${policy_conda_env}\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

echo -e "\033[32m[server] Launching policy_model_server (PID will be recorded)...\033[0m"
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
    --config /data/dex/RoboTwin/policy/GEACT/configs/ltx_model/policy_model_lerobot_robotwin.yaml \
    --weight /data/dex/Genie-Envisioner/GE_ACT_ROBOTWIN_FINETUNE/2025_12_17_04_19_20/step_9000/diffusion_pytorch_model.safetensors \
    --denoise_step 5 \
    --domain_name RoboTwin \
    --policy_conda_env ${policy_conda_env} \
    --gpu_id ${gpu_id} \
    --execution_step ${execution_step}&
SERVER_PID=$!

# Ensure the server is killed when this script exits
trap "echo -e '\033[31m[cleanup] Killing server (PID=${SERVER_PID})\033[0m'; kill ${SERVER_PID} 2>/dev/null" EXIT

conda deactivate
echo -e "\033[32m[client] Activating Conda environment: ${sim_conda_env}\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${sim_conda_env}"

# Start the client in the foreground
echo -e "\033[34m[client] Starting eval_policy_client on port ${port}...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_client.py \
    --config /data/dex/RoboTwin/policy/GEACT/deploy_policy.yml \
    --port ${port} \
    --overrides \
    --policy_name ${policy_name} \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --config /data/dex/RoboTwin/policy/GEACT/configs/ltx_model/policy_model_lerobot_robotwin.yaml \
    --weight /data/dex/Genie-Envisioner/GE_ACT_ROBOTWIN_FINETUNE/2025_12_17_04_19_20/step_9000/diffusion_pytorch_model.safetensors \
    --denoise_step 5 \
    --domain_name RoboTwin \
    --policy_conda_env ${policy_conda_env} \
    --gpu_id ${gpu_id} \
    --execution_step ${execution_step}

echo -e "\033[33m[main] eval_policy_client has finished; the server will be terminated.\033[0m"
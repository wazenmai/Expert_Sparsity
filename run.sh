# sudo pip3 uninstall transformer-engine

export HF_HUB_CACHE=/home/u2139934/Warehouse/cache/huggingface/hub/
export HF_HOME=/home/u2139934/Warehouse/cache/huggingface/
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# PRUNED_MODEL_PATH=/home/u2139934/MoE_Experiments/Expert_Sparsity/output/TinyLLama-4x1.1B-MoE_layerwise_pruning_r2_c4_128_20240619-235117
# MODEL_PATH=/home/u2139934/Warehouse/cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841
# PRUNED_MODEL_PATH=/home/u2139934/Warehouse/models/prune_baseline/TinyLLama-4x1.1B-MoE_layerwise_pruning_r1_c4_32_20240806-135915
PRUNED_MODEL_PATH=/home/u2139934/Warehouse/models/prune_baseline/Qwen1.5-MoE-A2.7B-Chat_qwen_layerwise_pruning_r45_c4_32_20240915-130518
MODEL_PATH=Qwen/Qwen1.5-MoE-A2.7B-Chat
OUTPUT_PATH=/home/u2139934/Warehouse/models/prune_baseline/Qwen1.5-MoE-A2.7B-Chat_qwen_layerwise_pruning_r45_c4_32_20241201-1939
# 
# --num_processes=1 \
# --mixed_precision=bf16 \
# --ipex \
# accelerate launch \
# -m 
# python3 main.py --method layerwise_pruning --r 1 --calib_set c4 --n_blocks_for_stat 32 --model_path $MODEL_PATH --output_path $OUTPUT_PATH |& tee log_tinyllama_r1_n32
python3 main.py --method qwen_layerwise_pruning --r 45 --calib_set c4 --n_blocks_for_stat 32 --model_path $MODEL_PATH --output_path $OUTPUT_PATH |& tee log_qwen_r45_10
# lm_eval --model hf \
#     --model_args pretrained=$OUTPUT_PATH,dtype=bfloat16,parallelize=True \
#     --tasks gsm8k --batch_size 8 |& tee log_result_qwen_45

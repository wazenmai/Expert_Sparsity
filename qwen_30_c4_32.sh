export HF_HUB_CACHE=/home/u2139934/Warehouse/cache/huggingface/hub/
export HF_HOME=/home/u2139934/Warehouse/cache/huggingface/
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# sudo pip uninstall --yes transformer-engine

MODEL_PATH=Qwen/Qwen1.5-MoE-A2.7B-Chat
OUTPUT_PATH=/home/u2139934/Warehouse/models/prune_baseline/Qwen1.5-MoE-A2.7B-Chat_qwen_layerwise_pruning_r30_c4_32_20241202-1154
# python3 main.py --method qwen_layerwise_pruning --r 30 --calib_set c4 --n_blocks_for_stat 32 --model_path $MODEL_PATH --output_path $OUTPUT_PATH |& tee new_results/log_qwen_r30_c4_32_gsm8k
lm_eval --model hf \
    --model_args pretrained=$OUTPUT_PATH,dtype=bfloat16,parallelize=True \
    --tasks gsm8k --batch_size 16 |& tee new_results/result_qwen_r30_c4_32_gsm8k
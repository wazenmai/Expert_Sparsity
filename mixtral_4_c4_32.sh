export HF_HUB_CACHE=/home/u2139934/Warehouse/cache/huggingface/hub/
export HF_HOME=/home/u2139934/Warehouse/cache/huggingface/
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

MODEL_PATH=/home/u2139934/Warehouse/cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841

OUTPUT_PATH=/home/u2139934/Warehouse/models/prune_baseline/Mixtral-8x7B-v0.1_mixtral_layerwise_pruning_r4_c4_32_20241202-1154
# python3 main.py --method layerwise_pruning --r 4 --calib_set c4 --n_blocks_for_stat 32 --model_path $MODEL_PATH --output_path $OUTPUT_PATH |& tee new_results/log_mixtral_r4_c4_32_gsm8k
lm_eval --model hf \
    --model_args pretrained=$OUTPUT_PATH,dtype=bfloat16,parallelize=True \
    --tasks gsm8k --batch_size 32 |& tee new_results/result_mixtral_r4_c4_32_gsm8k
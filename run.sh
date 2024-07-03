# sudo pip3 uninstall transformer-engine

HF_HUB_CACHE=/home/u2139934/Warehouse/cache/huggingface/hub/
HF_HOME=/home/u2139934/Warehouse/cache/huggingface/
PRUNED_MODEL_PATH=/home/u2139934/MoE_Experiments/Expert_Sparsity/output/TinyLLama-4x1.1B-MoE_layerwise_pruning_r2_c4_128_20240619-235117
MODEL_PATH=/home/u2139934/Warehouse/cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841
OUTPUT_PATH=/home/u2139934/Warehouse/models/prune_baseline/

# --num_processes=1 \
# --mixed_precision=bf16 \
# --ipex \
# accelerate launch \
# -m 
python3 main.py --method layerwise_pruning --r 6 --calib_set c4 --n_blocks_for_stat 32 --model_path $MODEL_PATH --output_path $OUTPUT_PATH |& tee log_last_layer
# lm_eval --model hf \
# --model_args pretrained=$PRUNED_MODEL_PATH,dtype=bfloat16,parallelize=True \
# --tasks winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte --batch_size 16 |& tee log2
import os
import os.path as osp
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import logging
import argparse
from argparse import Namespace
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from method import METHODS
from data import DATASETS, build_calib_loader
from eval.lm_eval import evaluate_fewshot


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=list(METHODS.keys()),
                        help=' '.join(['Supported pruning methods:'] + list(METHODS.keys())))
    parser.add_argument('--r', type=int, default=None,
                        help='Number of experts to preserve')
    parser.add_argument('--calib_set', type=str, required=True,
                        choices=list(DATASETS.keys()),
                        help=' '.join(['Supported calibration datasets:'] + list(DATASETS.keys())))
    parser.add_argument('--model_path', type=str, default=None, required=True,
                        help='Path to model to prune')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='Output path (pruned model, pruning results, etc.)')
    parser.add_argument('--max_block_size', type=int, default=2048,
                        help='Maximal sequence length of each sample in calibration set')
    parser.add_argument('--n_blocks_for_stat', type=int, default=128,
                        help='Number of sequences in calibration set. If set to 0 or negative, the whole dataset will be used')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for model inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers in dataloader')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproduction')
    parser.add_argument('--use_flash_attention_2', action='store_true',
                        help='If set, Flash Attention 2 will be used')

    return parser.parse_args()


def main(args: Namespace):
    print("GPUs: ", torch.cuda.device_count())
    logger.info(f'Arguments: {args}')

    if args.model_path.endswith('/'):
        args.model_path = args.model_path[:-1]
    model_name = args.model_path.split('/')[-1]

    if args.method.endswith('_pruning'):
        assert args.r is not None, 'Using pruning methods, argument `r` is required'
        save_path = osp.join(
            args.output_path, f'{model_name}_{args.method}_r{args.r}_{args.calib_set}_{args.n_blocks_for_stat}_{"fatt2_" if args.use_flash_attention_2 else ""}{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    else:
        if args.r is not None:
            logger.warn(f'Not using pruning methods, argument `r` is not used')
        save_path = osp.join(
            args.output_path, f'{model_name}_{args.method}_{args.calib_set}_{args.n_blocks_for_stat}_{"fatt2_" if args.use_flash_attention_2 else ""}{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    logger.info(f'Save path: {save_path}')
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 else None
    )

    calib_loader = build_calib_loader(args.calib_set, tokenizer, args.max_block_size,
                                      args.n_blocks_for_stat, args.batch_size, args.num_workers, args.seed)
    
    """
    # Last Layer Experiment
    layer = model.model.layers[31].block_sparse_moe

    experts_to_reserve = [0, 1, 2, 3, 5, 7]
    gate_new = torch.nn.Linear(in_features=layer.gate.in_features,
                                out_features=args.r, bias=False, device='cpu', dtype=torch.bfloat16)
    gate_new.weight.data = layer.gate.weight.data[list(
        experts_to_reserve)]
    layer.gate = gate_new

    layer.experts = torch.nn.ModuleList(
        [layer.experts[i] for i in experts_to_reserve])
    layer.num_experts = args.r

    print(model.model.layers[31].block_sparse_moe)
    """

    model, info = METHODS[args.method](model, calib_loader, args)
    if args.r < model.config.num_experts_per_tok:
        for layer in model.model.layers:
            moe = layer.block_sparse_moe
            layer.top_k = args.r
            print(layer.top_k)

    os.makedirs(save_path, exist_ok=False)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save((args, info), osp.join(save_path, 'pruning_info.pt'))


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # task = ["winogrande", "arc_challenge", "arc_easy", "boolq", "hellaswag", "mmlu", "openbookqa", "rte"]
    # eval_batch_size = [32, 32, 32, 32, 32, 32, 32, 32]
    # task = ["openbookqa", "rte"]
    # eval_batch_size = [64, 64]
    # for i, t in enumerate(task):
    #     evaluate_fewshot(
    #         model, tokenizer=tokenizer, task=t, num_fewshot=0, eval_batch_size=eval_batch_size[i], log=True
    #     )


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    args = parse_args()
    main(args)

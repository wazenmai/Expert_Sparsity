from tqdm import tqdm
from argparse import Namespace
import logging

import torch
from torch.utils.data import DataLoader
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM

from model import PrunableMixtralSparseMoeBlockWrapper, PrunableQwenSparseMoeBlockWrapper


logger = logging.getLogger(__name__)


def layerwise_pruning_qwen(model: Qwen2MoeForCausalLM, calib_loader: DataLoader, args: Namespace):
    original_devices = {}
    for l, layer in enumerate(model.model.layers):
        original_devices[l] = next(layer.mlp.parameters()).device

    for l, layer in enumerate(model.model.layers):
        layer.mlp = PrunableQwenSparseMoeBlockWrapper(
            layer.mlp, r=args.r)
        layer.mlp.cache_X = True
        layer.mlp.cache_Z = True
    
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...')):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    logger.info('Moving whole model to cpu...')
    # model.to('cpu')
    for l, layer in enumerate(model.model.layers):
        layer = layer.to('cpu')
    torch.cuda.empty_cache()

    global_loss_history = dict()
    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Enumerating loss on sample set...'):
        print("layer ", l)
        b = layer.mlp
        if not hasattr(b, 'cache_space'):
            continue
        b.to(original_devices[l])
        print("model device {}".format(b.model.gate.weight.data.device))
        loss_history = b.enumerate()
        global_loss_history[l] = loss_history
        b.prune()
        b.to('cpu')

    logger.info('Merging & saving...')
    for l, layer in enumerate(model.model.layers):
        layer.mlp = layer.mlp.model
        layer = layer.to(original_devices[l])

    model.num_experts = args.r
    model.config.num_experts = args.r

    return model, (global_loss_history, )


def layerwise_pruning(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    original_devices = {}
    for l, layer in enumerate(model.model.layers):
        original_devices[l] = next(layer.block_sparse_moe.parameters()).device
    
    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe, r=args.r)
        layer.block_sparse_moe.cache_X = True
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...')):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    logger.info('Moving whole model to cpu...')
    model.to('cpu')
    # for l, layer in enumerate(model.model.layers):
    #     layer = layer.to('cpu')
    torch.cuda.empty_cache()

    global_loss_history = dict()
    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Enumerating loss on sample set...'):
        print("layer ", l)
        b = layer.block_sparse_moe
        if not hasattr(b, 'cache_space'):
            continue
        b.to(original_devices[l])
        loss_history = b.enumerate()
        global_loss_history[l] = loss_history
        b.prune()
        b.to('cpu')

    logger.info('Merging & saving...')
    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = layer.block_sparse_moe.model
        # layer = layer.to(original_devices[l])

    model.num_experts = args.r
    model.config.num_local_experts = args.r

    return model, (global_loss_history, )


def progressive_pruning(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe, r=args.r)
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Computing Z activations on sample set...')):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    del model_inputs
    del outputs
    torch.cuda.empty_cache()

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe.cache_Z = False

    # Drop
    global_loss_history = dict()

    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Dropping layers...'):
        b = layer.block_sparse_moe

        b.cache_X = True
        with torch.inference_mode():
            for i, batch in enumerate(calib_loader):
                model_inputs = model.prepare_inputs_for_generation(**batch)
                outputs = model(**model_inputs)
                assert outputs is not None

        del model_inputs
        del outputs
        torch.cuda.empty_cache()
        b.cache_X = False

        loss_history = b.enumerate()
        global_loss_history[l] = loss_history

        b.prune()
        layer.block_sparse_moe = b.model

    # Prune & save
    model.num_experts = args.r
    model.config.num_local_experts = args.r

    return model, (global_loss_history, )

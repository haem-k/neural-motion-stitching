import argparse
import numpy as np
import random
from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nms.training.dataset import VQVAEDataset, MGDataset
import nms.utils.utils as utils

'''
General
'''
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_ckpt(ckpt_dir: str, ckpt_name: str):
    if ckpt_name == "":
        ckpt_name_list = utils.list_files_in_dir(ckpt_dir)
        ckpt_name = ckpt_name_list[-1]
    else:
        ckpt_name = f"{ckpt_name}.pt"
    load_ckpt_dir = f"{ckpt_dir}/{ckpt_name}"
    ckpt = torch.load(load_ckpt_dir)
    return ckpt, ckpt_name


'''
VQVAE
'''
def prepare_vqvae_dataset(
    train_path: str,
    val_path: str,
    batch_size: int,
    seed: int
) -> Tuple[Dict[str, DataLoader], torch.Tensor]:

    train_dataset = VQVAEDataset(train_path)
    val_dataset = VQVAEDataset(val_path)

    loaders = {}
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    loaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=train_generator)
    loaders["val"] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return loaders

def load_model(
    model: nn.Module,
    model_dir: str,
    ckpt_name: str = "",
    freeze: bool = True
):
    ckpt_dir = f"{model_dir}/checkpoint"
    ckpt, ckpt_name = load_ckpt(ckpt_dir, ckpt_name) # get ckpt with latest epoch
    model.load_state_dict(ckpt["model_state_dict"])

    # freeze
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model, ckpt_name

def save_model(
    model: nn.Module,
    config: argparse.Namespace,
    epoch: int,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss: float
):
    result_dir = f"{config.result_dir}/{config.exp_name}"
    ckpt_dir = f"{result_dir}/checkpoint"
    utils.check_create_dir(ckpt_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, f"{ckpt_dir}/epoch_{epoch}.pt")
    print(f"Saved {epoch}th trained checkpoint")
    return model


'''
Motion Generator
'''

def prepare_mg_dataset(
    train_dir: str,
    val_dir: str,
    config: argparse.Namespace,
) -> Dict[str, DataLoader]:

    # init datasets
    train_dataset = MGDataset(train_dir, config.archi.max_len, config.block_nof, mode="train", lengths=config.lengths)
    val_dataset = MGDataset(val_dir, config.archi.max_len, config.block_nof, mode="val", lengths=config.lengths)

    loaders = {}
    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)
    loaders["train"] = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=train_generator, num_workers=config.num_workers, pin_memory=True)
    loaders["val"] = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    return loaders

def get_square_padding_mask(max_len, pred_end_idx, n_head, batch_size, device=None):
    '''
    Compute mask for padding data with various lengths
    Value should be True if masking is needed

    max_len: max len for motion code sequence
    pred_end_cidx: [64]

    Returns:
        padding_mask: [batch_size, n_head, max_len, max_len]
    '''
    seq_range = torch.arange(max_len).unsqueeze(0)
    padding_mask_col = (seq_range >= pred_end_idx.unsqueeze(1)).unsqueeze(-1)
    padding_mask_row = padding_mask_col.transpose(-1, -2)

    padding_mask_col = padding_mask_col.unsqueeze(1)
    padding_mask_row = padding_mask_row.unsqueeze(1)
    # make squared mask so that rows can be ignored too

    padding_mask_col = padding_mask_col.expand(-1, n_head, -1, max_len)
    padding_mask_row = padding_mask_row.expand(-1, n_head, max_len, -1)
    padding_mask = torch.bitwise_or(padding_mask_col, padding_mask_row)
    if device is not None:
        padding_mask = padding_mask.to(device)
    return padding_mask


def get_disc_mask(
    max_len: int,
    valid_nof: int,
    block_nof: int,
    device: torch.device
):
    half = block_nof // 2
    n_valid_block_max = int(max_len/block_nof) - 1 # two half blocks will be trimmed

    disc_valid_idx = (valid_nof - block_nof - half) // block_nof
    range_idx = torch.arange(n_valid_block_max, device=valid_nof.device).unsqueeze(0)
    disc_loss_mask = range_idx < disc_valid_idx.unsqueeze(1)

    true_pad = torch.ones_like(disc_loss_mask[:, :1], dtype=torch.bool) # one valid block will be appended at the end
    disc_loss_mask = torch.cat([disc_loss_mask, true_pad], dim=1).reshape([-1, 1])
    disc_loss_mask = disc_loss_mask.to(torch.float32).to(device)
    return disc_loss_mask
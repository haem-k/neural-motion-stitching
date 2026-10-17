import torch
import sys
import argparse
import numpy as np
import nms.utils.utils as utils
import nms.utils.network_utils as utils_n
from nms.training.lr import CosineAnnealingWarmUpRestarts
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import nms.training.loss as loss
from nms.modules.discriminator import Discriminator
from nms.modules.motion_generator import MotionGenerator, get_inbetween_mask, get_square_padding_mask
from nms.motion.motion_op_tensor import fk

def get_fake_input(
    pred_world_pos: torch.Tensor,
    targetB_world_pos: torch.Tensor,
    jnt_map: torch.Tensor,
    block_nof: int,
    valid_nof: torch.Tensor,
):
    '''
    Get fake input for discriminator training.
    The logic should match how the loss is masked in get_disc_mask
    
    Output:
        fake_input: torch.Tensor
    '''
    
    # prepare index info for various lengths
    batch_size = pred_world_pos.shape[0]
    half = block_nof // 2
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand(batch_size, -1)
    half_offset = torch.arange(half).to(pred_world_pos.device)
    pred_last_half_idx = (valid_nof.unsqueeze(1) - block_nof - half) + half_offset
    
    # get all joints' trans disp: targetA + output 
    pred_trans_disp = pred_world_pos[:, 1:, jnt_map] - pred_world_pos[:, :-1, jnt_map] # [batch_size, max_len-1, 22, 3]
    pred_trans_disp_trim = pred_trans_disp[:, half:-half+1] # -half+1 because trans disp is one frame short [batch_size, max_len-2*half, 22, 3]
    pred_trans_disp_batch = pred_trans_disp_trim.reshape([batch_size, -1, block_nof, pred_trans_disp_trim.shape[-2] * pred_trans_disp_trim.shape[-1]])

    # get all joints' trans disp: output + targetB
    pred_last_half_world_pos = pred_world_pos[batch_idx, pred_last_half_idx]
    pred_last_half_world_pos = pred_last_half_world_pos[..., jnt_map, :]
    targetB_first_half_world_pos = targetB_world_pos[batch_idx, :half+1].squeeze() # half+1 for full disp info
    targetB_first_half_world_pos = targetB_first_half_world_pos[..., jnt_map, :]
    
    pred_last_world_pos = torch.cat([pred_last_half_world_pos, targetB_first_half_world_pos], dim=1)
    pred_last_trans_disp = pred_last_world_pos[:, 1:] - pred_last_world_pos[:, :-1] # [batch_size, block_nof, 22, 3]
    pred_last_trans_disp_batch = pred_last_trans_disp.reshape([batch_size, 1, block_nof, -1])

    # get adv loss with discriminator
    fake_input = torch.cat([pred_trans_disp_batch, pred_last_trans_disp_batch], dim=-3)
    fake_input = fake_input.reshape([-1, fake_input.shape[-2], fake_input.shape[-1]])
    return fake_input

def train_mg(config: argparse.Namespace):
    print("[Training Motion Generator]")
    
    # setup training
    tensorboard_dir = f"{config.tensorboard_dir}/{config.exp_name}"
    writer = SummaryWriter(tensorboard_dir)
    utils.check_create_dir(tensorboard_dir)
    utils_n.set_seeds(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(f"Training setup, using device: {device}")
    
    # setup motion generation
    mogen = MotionGenerator(config, device)
    config = mogen.config # retrieve updated config
    
    # prepare dataset
    print(f"Preparing dataset...")
    data_dir = f"{config.dataset_dir}/{config.dataset_name}"
    dataloaders = utils_n.prepare_mg_dataset(
        f"{data_dir}/train",
        f"{data_dir}/val",
        config
    )
    n_train = len(dataloaders["train"].dataset)
    n_val = len(dataloaders["val"].dataset)
    print(f"Prepared dataset: {n_train} training / {n_val} validation")
    
    # init training params 
    lr = config.lr
    total_iteration = len(dataloaders["train"]) * config.epoch
    iter_warmup = int(total_iteration * config.iter_warmup_ratio)
    optim = torch.optim.AdamW(mogen.transformer.parameters(), lr=0.0, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(optim, T_0=total_iteration, T_mult=config.T_mult, eta_max=config.lr, T_up=iter_warmup, gamma=config.gamma)

    # prepare discriminator
    config.disc.block_nof = config.block_nof
    disc_stats_path = f"{data_dir}/statistics_disc.pt"
    disc = Discriminator(config.disc, mogen.skel.get_jname_list(), disc_stats_path).to(device)
    disc_iteration = total_iteration - iter_warmup
    optim_disc = torch.optim.Adam(disc.parameters(), lr=config.disc.lr_disc, betas=(0.5, 0.9))
    scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(optim_disc, disc_iteration, config.disc.lr_disc * 0.01)
        
    # save updated config and datareps
    utils.save_config(config, f"{config.result_dir}/{config.exp_name}", "mg_train")
    mogen.save_datarep(f"{config.result_dir}/{config.exp_name}")
    
    # create checkpoint directory for newly trained weights
    ckpt_dir = f"{config.result_dir}/{config.exp_name}/checkpoint"
    utils.check_create_dir(ckpt_dir)
    epoch = 0
    iteration = 0
    loss_weight = {
        "loss_trans": config.w_trans,
        "loss_fk": config.w_fk,
        "loss_dest": config.w_dest,
        "loss_adv": config.w_adv
    }
    loss_names = [
        "loss",
        "loss_trans",
        "loss_fk",
        "loss_dest",
        "loss_adv"
    ]
    disc_loss_names = [
        "loss_disc",
        "loss_fake",
        "loss_real",
        "grad_penalty"
    ]
    log_names = loss_names + ["learning rate"]
    disc_log_names = disc_loss_names + ["learning rate"]
    
    epoch_bar = tqdm(range(epoch, config.epoch), position=0)
    epoch_bar.set_description("Epoch")
    max_len = config.archi.max_len
    block_nof = config.block_nof
    for epoch in epoch_bar:
        mogen.transformer.train()
        disc.train()
        train_log_values = {name: 0.0 for name in log_names}
        train_disc_log_values = {name: 0.0 for name in disc_log_names}
        
        for batch_idx, (_, valid_nof, gt, input) in enumerate(dataloaders["train"]):
            optim.zero_grad()
            optim_disc.zero_grad()
            iteration += 1
            lr = optim.param_groups[0]['lr']

            # attention mask for length-varying input
            padding_mask = get_square_padding_mask(max_len, valid_nof, config.archi.n_head, device)
            
            # forward, data un-normalized in this scope
            gt = gt.to(device)
            input = input.to(device)
            out = mogen.transformer(input, padding_mask)
            
            # mask for loss
            padding_mask = ~padding_mask[..., 0, 0, :] # length-varying
            pred_start_idx = torch.ones_like(valid_nof) * block_nof
            pred_end_idx = valid_nof - block_nof
            inbetween_mask = get_inbetween_mask(max_len, pred_start_idx, pred_end_idx, device)
            valid_nof = valid_nof.to(device)

            # align output
            local_trfs = mogen.get_aligned_local_trfs(input, out, valid_nof, padding_mask=padding_mask, gt=gt)
            targetB_world_trf = fk(local_trfs["targetB"], mogen.skel_offsets, mogen.skel_parent_indices)
            pred_world_trf = fk(local_trfs["pred"], mogen.skel_offsets, mogen.skel_parent_indices)
            targetB_world_pos = targetB_world_trf[..., :3, 3]
            pred_world_pos = pred_world_trf[..., :3, 3]
            
            with torch.no_grad():
                gt_world_trf = fk(local_trfs["gt"], mogen.skel_offsets, mogen.skel_parent_indices)
                gt_world_pos = gt_world_trf[..., :3, 3]
            
            del targetB_world_trf, pred_world_trf, gt_world_trf
            
            # for disc inference
            disc_loss_mask = utils_n.get_disc_mask(max_len, valid_nof, block_nof, device)
            disc_fake_input = get_fake_input(pred_world_pos, targetB_world_pos, disc.jnt_map, block_nof, valid_nof)
            
            # loss for transformer
            loss_dict = {name: torch.tensor(0).to(device) for name in loss_names}
            loss_dict["loss_trans"] = loss.trans_loss(out, gt, inbetween_mask)
            loss_dict["loss_fk"] = loss.fk_loss(pred_world_pos, gt_world_pos, inbetween_mask)
            if config.use_dest:
                loss_dict["loss_dest"] = loss.dest_loss(out, gt, mogen.gt_feat_start_idx, mogen.gt_feat_end_idx, valid_nof, block_nof)
            if config.use_adv:
                if iteration >= iter_warmup: # start using adv loss only after warmup
                    loss_dict["loss_adv"] = loss.adv_loss(disc, disc_fake_input, disc_loss_mask)
                    
            #   - combine loss
            for i in range(len(loss_names[1:])):
                loss_dict["loss"] = loss_dict["loss"] + loss_dict[loss_names[i+1]] * loss_weight[loss_names[i+1]]
            loss_total = loss_dict["loss"]
            loss_total.backward()
            optim.step()
            scheduler.step() # update transformer scheduler every iteration, not epoch
            
            #   - log values
            for key in loss_dict:
                train_log_values[key] = loss_dict[key].item()
            train_log_values["learning rate"] = lr
            
            for i in range(len(log_names)):
                writer.add_scalar(f"Train/{log_names[i]}", train_log_values[log_names[i]], iteration)
            
            if config.use_adv:
                # loss for discriminator
                if iteration >= iter_warmup:
                    disc_loss_dict = loss.disc_loss(
                        gt_world_pos,
                        disc,
                        disc_fake_input,
                        disc_loss_mask,
                        block_nof
                    )
                    loss_disc = disc_loss_dict["loss_disc"]
                    loss_disc.backward()
                    optim_disc.step()
                    disc_lr = optim_disc.param_groups[0]["lr"]

                    #   - log values for disc
                    for key in disc_loss_dict:
                        train_disc_log_values[key] = disc_loss_dict[key].item()
                    train_disc_log_values["learning rate"] = disc_lr
                    
                    for i in range(len(disc_log_names)):
                        writer.add_scalar(f"Train Disc/{disc_log_names[i]}", train_disc_log_values[disc_log_names[i]], iteration)
                    scheduler_disc.step()
            
        # validation
        mogen.transformer.eval()
        disc.eval()
        val_log_values = {name: 0.0 for name in log_names[:-1]}
        val_disc_log_values = {name: 0.0 for name in disc_log_names[:-1]}
        for batch_idx, (_, valid_nof, gt, input) in enumerate(dataloaders["val"]):
            with torch.no_grad():       
                # attention mask for length-varying input
                padding_mask = get_square_padding_mask(max_len, valid_nof, config.archi.n_head, device)
                
                # forward, data un-normalized in this scope
                gt = gt.to(device)
                input = input.to(device)
                
                out = mogen.transformer(input, padding_mask)
                
                # compute mask
                pred_start_idx = torch.ones_like(valid_nof) * block_nof
                pred_end_idx = valid_nof - block_nof
                padding_mask = ~padding_mask[..., 0, 0, :]
                inbetween_mask = get_inbetween_mask(max_len, pred_start_idx, pred_end_idx, device)
                valid_nof = valid_nof.to(device)

                local_trfs = mogen.get_aligned_local_trfs(input, out, valid_nof, padding_mask=padding_mask, gt=gt)
                targetB_world_trf = fk(local_trfs["targetB"], mogen.skel_offsets, mogen.skel_parent_indices)
                pred_world_trf = fk(local_trfs["pred"], mogen.skel_offsets, mogen.skel_parent_indices)
                gt_world_trf = fk(local_trfs["gt"], mogen.skel_offsets, mogen.skel_parent_indices)
                
                targetB_world_pos = targetB_world_trf[..., :3, 3]
                pred_world_pos = pred_world_trf[..., :3, 3]
                gt_world_pos = gt_world_trf[..., :3, 3]
                
                # for disc inference
                disc_loss_mask = utils_n.get_disc_mask(max_len, valid_nof, block_nof, device)
                disc_fake_input = get_fake_input(pred_world_pos, targetB_world_pos, disc.jnt_map, block_nof, valid_nof)
                
                # loss for transformer
                loss_dict = {name: torch.tensor(0).to(device) for name in loss_names}
                loss_dict["loss_trans"] = loss.trans_loss(out, gt, inbetween_mask)
                loss_dict["loss_fk"] = loss.fk_loss(pred_world_pos, gt_world_pos, inbetween_mask)
                if config.use_dest:
                    loss_dict["loss_dest"] = loss.dest_loss(out, gt, mogen.gt_feat_start_idx, mogen.gt_feat_end_idx, valid_nof, block_nof)
                if config.use_adv:
                    # start using adv loss only after warmup
                    if iteration >= iter_warmup: 
                        loss_dict["loss_adv"] = loss.adv_loss(disc, disc_fake_input, disc_loss_mask)
                        
                #   - combine loss
                for i in range(len(loss_names[1:])):
                    loss_dict["loss"] = loss_dict["loss"] + loss_dict[loss_names[i+1]] * loss_weight[loss_names[i+1]]
                loss_total = loss_dict["loss"]
                
                if config.use_adv:
                    # loss for discriminator
                    if iteration >= iter_warmup:
                        disc_loss_dict = loss.disc_loss(
                            gt_world_pos,
                            disc,
                            disc_fake_input,
                            disc_loss_mask,
                            block_nof,
                            "val"
                        )

                # log values
                for key in loss_dict:
                    val_log_values[key] += loss_dict[key].item()
                    
                # log values for disc
                if config.use_adv:
                    if iteration >= iter_warmup:
                        for key in disc_loss_dict:
                            val_disc_log_values[key] += disc_loss_dict[key].item()
                                    
        for i in range(len(log_names[:-1])):
            writer.add_scalar(f"Val/{log_names[i]}", val_log_values[log_names[i]]/len(dataloaders["val"]), epoch)
        if config.use_adv:
            for i in range(len(disc_log_names[:-1])):
                writer.add_scalar(f"Val Disc/{disc_log_names[i]}", val_disc_log_values[disc_log_names[i]]/len(dataloaders["val"]), epoch)
        
        # Save checkpoint if needed
        if epoch % config.save_ckpt_freq == 0:
            utils_n.save_model(mogen.transformer, config, epoch, optim, scheduler, train_log_values["loss"])
        
    # Training done, save checkpoint
    writer.close()
    utils_n.save_model(mogen.transformer, config, epoch, optim, scheduler, train_log_values["loss"])
    print(f"Training done.")
    
    
if __name__ == "__main__":
    '''
    Train Motion Generator. 
    Statistics for the dataset needs to be precomputed and saved as .pt
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="mg_train")
    args = parser.parse_args()
    config = utils.load_config("./nms/config", args.config)
    
    if config.exp_name is None:
        print("Please provide exp_name")
        sys.exit()
    
    utils.print_argument(config)
    train_mg(config)
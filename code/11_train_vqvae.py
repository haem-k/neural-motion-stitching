import torch
import logging
import argparse
import sys
import numpy as np
import nms.utils.utils as utils
import nms.utils.network_utils as utils_n
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from nms.rotation.rotation_tensor import ortho6d_to_rotmat, rotmat_to_trf, pos_to_trf
from nms.training.process import recon_pos_from_disp_t
from nms.modules.cvqvae import CVQVAE

def compute_recon_loss(
    out: torch.Tensor,
    x: torch.Tensor,
    feat_start_idx: dict,
    feat_end_idx: dict,
    loss_weight: dict
):
    loss_dict = {}
    loss_recon = torch.zeros([1]).to(out.device)
    for key in feat_start_idx:
        out_val = out[..., feat_start_idx[key]:feat_end_idx[key]]
        x_val = x[..., feat_start_idx[key]:feat_end_idx[key]]
        loss = F.mse_loss(out_val, x_val)
        loss_dict[key] = loss
        w_loss = loss_weight[key]
        loss_recon = loss_recon + loss * w_loss
    
    loss_dict["loss_recon"] = loss_recon
    return loss_dict
    
    
def compute_pos_loss(
    out: torch.Tensor,
    x: torch.Tensor,
    feat_start_idx: dict,
    feat_end_idx: dict
):
    # Recover world joint position
    # root orient
    out_root_orient = out[..., feat_start_idx["root_orient"]:feat_end_idx["root_orient"]].unsqueeze(-2)
    out_root_orient = ortho6d_to_rotmat(out_root_orient)
    x_root_orient = x[..., feat_start_idx["root_orient"]:feat_end_idx["root_orient"]].unsqueeze(-2)
    x_root_orient = ortho6d_to_rotmat(x_root_orient)
    
    # accum root trans disp for x, z position
    pad = (1, 1)
    out_root_trans_disp = out[..., feat_start_idx["root_trans_disp"]:feat_end_idx["root_trans_disp"]].unsqueeze(-2)
    out_root_h = out[..., feat_start_idx["root_height"]:feat_end_idx["root_height"]].unsqueeze(-2)
    out_root_h_init = out_root_h[..., :1, :, :]
    out_root_pos_init = F.pad(out_root_h_init, pad)
    out_root_pos_accum = recon_pos_from_disp_t(out_root_trans_disp, out_root_pos_init)
    out_root_pos = torch.cat([out_root_pos_accum[..., :-1, :, 0:1], out_root_h, out_root_pos_accum[..., :-1, :, 2:3]], dim=-1)
    
    x_root_trans_disp = x[..., feat_start_idx["root_trans_disp"]:feat_end_idx["root_trans_disp"]].unsqueeze(-2)
    x_root_h = x[..., feat_start_idx["root_height"]:feat_end_idx["root_height"]].unsqueeze(-2)
    x_root_h_init = x_root_h[..., :1, :, :]
    x_root_pos_init = F.pad(x_root_h_init, pad)
    x_root_pos_accum = recon_pos_from_disp_t(x_root_trans_disp, x_root_pos_init)
    x_root_pos = torch.cat([x_root_pos_accum[..., :-1, :, 0:1], x_root_h, x_root_pos_accum[..., :-1, :, 2:3]], dim=-1)
    
    # reconstruct root trf
    out_root_trf = rotmat_to_trf(out_root_orient, out_root_pos)
    x_root_trf = rotmat_to_trf(x_root_orient, x_root_pos)
    
    # get root relative joint positions
    out_joint_pos_root_rel = out[..., feat_start_idx["joint_pos_root_rel"]:feat_end_idx["joint_pos_root_rel"]]
    out_joint_pos_root_rel = out_joint_pos_root_rel.reshape([*out_joint_pos_root_rel.shape[:-1], -1, 3])
    out_joint_pos_root_rel = pos_to_trf(out_joint_pos_root_rel)
    
    x_joint_pos_root_rel = x[..., feat_start_idx["joint_pos_root_rel"]:feat_end_idx["joint_pos_root_rel"]]
    x_joint_pos_root_rel = x_joint_pos_root_rel.reshape([*x_joint_pos_root_rel.shape[:-1], -1, 3])
    x_joint_pos_root_rel = pos_to_trf(x_joint_pos_root_rel)
    
    # reconstruct world position of joints
    out_world_pos_trf = torch.matmul(out_root_trf, out_joint_pos_root_rel)
    x_world_pos_trf = torch.matmul(x_root_trf, x_joint_pos_root_rel)
    
    out_world_pos = out_world_pos_trf[..., :3, 3]
    x_world_pos = x_world_pos_trf[..., :3, 3]
    
    loss_pos = F.mse_loss(out_world_pos, x_world_pos)
    return loss_pos
    

def train_vqvae(config: argparse.Namespace):
    print("[Training VQVAE]")

    # setup training
    result_dir = f"{config.result_dir}/{config.exp_name}"
    utils.check_create_dir(result_dir)
    
    tensorboard_dir = f"{config.tensorboard_dir}/{config.exp_name}"
    utils.check_create_dir(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    
    utils_n.set_seeds(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(f"Training setup, using device: {device}")

    # prepare dataset
    print(f"Preparing dataset...")
    data_dir = f"{config.dataset_dir}/{config.dataset_name}"
    train_datarep = utils.load_config(data_dir, config.datarep_name, True)
    total_dim, feat_start_idx, feat_end_idx, _, _ = utils.get_feature_dims(train_datarep)
    dataloaders = utils_n.prepare_vqvae_dataset(
        f"{data_dir}/train_data.pt",
        f"{data_dir}/val_data.pt",
        batch_size=config.batch_size,
        seed=config.seed
    )
    train_data_shape = dataloaders["train"].dataset.get_data_shape()
    mean, std = dataloaders["train"].dataset.get_mean_std()
    statistics = {}
    statistics["mean"] = mean
    statistics["std"] = std
    print(f"Prepared dataset, total training dataset shape: {train_data_shape}")

    # prepare network model
    config.archi.input_len = train_data_shape[1] # length of motion block
    config.archi.input_dim = total_dim
    model = CVQVAE(config.archi, statistics=statistics).to(device)
    
    print("Initialized network model")
    utils.print_model(model)
    utils.save_config(train_datarep, f"{config.result_dir}/{config.exp_name}", "vqvae_datarep")
    utils.save_config(config, f"{config.result_dir}/{config.exp_name}", "vqvae_train")
    
    # init training params
    epoch = 0
    lr = config.lr
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, config.epoch, config.lr * 0.01)
    
        
    # hyperparams
    recon_loss_weight = {
        "root_height": config.w_root_height,
        "root_orient": config.w_root_orient,
        "root_trans_disp": config.w_root_trans_disp,
        "root_ang_disp": config.w_root_ang_disp,
        "joint_pos_root_rel": config.w_joint_pos_root_rel,
        "joint_trans_disp": config.w_joint_trans_disp
    }
    
    # training loop
    log_names = ["Total loss", "Recon loss", "Position loss", "VQ loss", "Contras loss", "Perplexity", "Learning rate"]
    epoch_bar = tqdm(range(epoch, config.epoch), position=0)
    epoch_bar.set_description("Epoch")
    for epoch in epoch_bar:
        # training
        model.train()
        train_log_values = {name: 0.0 for name in log_names}

        # forward
        for _, x in dataloaders["train"]:
            optim.zero_grad()
            
            x = x.to(device)
            out, code_idx, loss_vq, loss_contras, perplexity = model(x) # commit loss, contras loss already mutliplied with weight
            loss_pos = compute_pos_loss(out, x, feat_start_idx, feat_end_idx)
            
            out_norm = (out - model.mean) / model.std
            x_norm = (x - model.mean) / model.std
            loss_recon_dict = compute_recon_loss(out_norm, x_norm, feat_start_idx, feat_end_idx, recon_loss_weight)
            loss_recon = loss_recon_dict["loss_recon"]

            loss = config.w_recon * loss_recon + config.w_pos * loss_pos + loss_vq + loss_contras
            loss.backward()
            optim.step()

            train_log_values["Total loss"] += loss.item()
            train_log_values["Recon loss"] += loss_recon.item()
            train_log_values["Position loss"] += loss_pos.item()
            train_log_values["VQ loss"] += loss_vq.item()
            train_log_values["Contras loss"] += loss_contras.item()
            train_log_values["Perplexity"] += perplexity.item()
        
        train_log_values["Learning rate"] = lr # recode every epoch
        lr = optim.param_groups[0]['lr']
        scheduler.step()

        # record
        for i in range(len(log_names)):
            if i >= len(log_names) - 1:
                writer.add_scalar(f"Train/{log_names[i]}", train_log_values[log_names[i]], epoch)
            else:
                writer.add_scalar(f"Train/{log_names[i]}", train_log_values[log_names[i]]/len(dataloaders["train"]), epoch)


        # val
        model.eval()
        val_log_values = {name: 0.0 for name in log_names}
        for _, x in dataloaders["val"]:
            with torch.no_grad():
                x = x.to(device)
                out, code_idx, loss_vq, loss_contras, perplexity = model(x) # commit loss, contras loss already mutliplied with weight
                loss_pos = compute_pos_loss(out, x, feat_start_idx, feat_end_idx)
                
                out_norm = (out - model.mean) / model.std
                x_norm = (x - model.mean) / model.std
                loss_recon_dict = compute_recon_loss(out_norm, x_norm, feat_start_idx, feat_end_idx, recon_loss_weight)
                loss_recon = loss_recon_dict["loss_recon"]

                loss = config.w_recon * loss_recon + config.w_pos * loss_pos + loss_vq + loss_contras

                val_log_values["Total loss"] += loss.item()
                val_log_values["Recon loss"] += loss_recon.item()
                val_log_values["Position loss"] += loss_pos.item()
                val_log_values["VQ loss"] += loss_vq.item()
                val_log_values["Contras loss"] += loss_contras.item()
                val_log_values["Perplexity"] += perplexity.item()
                
        for i in range(len(log_names) - 1):
            writer.add_scalar(f"Val/{log_names[i]}", val_log_values[log_names[i]]/len(dataloaders["val"]), epoch)
        
        # Save checkpoint if needed
        if epoch != 0 and epoch % config.save_ckpt_freq == 0:
            utils_n.save_model(model, config, epoch, optim, scheduler, train_log_values["Total loss"])
    
    # Training done, save checkpoint
    writer.close()
    model.save_codebook(result_dir, train_datarep) # save codebook
    utils_n.save_model(model, config, epoch, optim, scheduler, train_log_values["Total loss"])
    print(f"Training done.")
    
    # Export all validation result
    if config.val_export:
        val_midx = []
        val_input = []
        val_output = []
        val_code = []
        val_code_idx = []
        for i, x in dataloaders["val"]:
            model.eval()
            with torch.no_grad():
                x = x.to(device)
                out, code_idx, loss_vq, loss_contras, perplexity = model(x)
                
                x = x.cpu().numpy()
                out = out.cpu().numpy()
                code_idx = code_idx.cpu().numpy() 
                
                val_midx.append(i)
                val_input.append(x)
                val_output.append(out)
                val_code_idx.append(code_idx)
        
        val_midx = np.concatenate(val_midx, axis=0)
        val_input = np.concatenate(val_input, axis=0)
        val_output = np.concatenate(val_output, axis=0)
        val_code_idx = np.concatenate(val_code_idx, axis=0)

        np.save(f"{config.result_dir}/{config.exp_name}/val_midx.npy", val_midx)
        np.save(f"{config.result_dir}/{config.exp_name}/val_input.npy", val_input)
        np.save(f"{config.result_dir}/{config.exp_name}/val_output.npy", val_output)
        np.save(f"{config.result_dir}/{config.exp_name}/val_code_idx.npy", val_code_idx)
        
        if len(val_code) != 0:
            # Export codes for all validation
            val_code = np.concatenate(val_code, axis=0)
            np.save(f"{config.result_dir}/{config.exp_name}/val_code.npy", val_code)
            
        print("\nExported validation results")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="vqvae_train")
    args = parser.parse_args()
    config = utils.load_config("./nms/config", args.config)
    
    if config.exp_name is None:
        print("Please provide exp_name")
        sys.exit()
    
    utils.print_argument(config)
    train_vqvae(config)
    
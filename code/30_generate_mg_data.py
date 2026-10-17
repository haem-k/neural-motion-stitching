import numpy as np
import argparse
import shutil
import torch
import sys
from pathlib import Path
import nms.utils.bvh as bvh
import nms.utils.utils as utils
import nms.utils.network_utils as utils_n
from nms.training.process import interp_guide, motion_to_feat
from nms.motion.motion_op_numpy import transform_motion_to, fk, get_joint_trans_disp
from tqdm.auto import tqdm
from nms.motion.motion import Skeleton
from nms.modules.cluster import Clusters
from nms.modules.graph import ClusterTransitionGraph
import nms.modules.preprocess as preprocess

def motions_to_samples(
    motions: list,
    clusters: Clusters,
    cluster_graph: ClusterTransitionGraph,
    n_block: int,
    target_datarep: dict,
    guide_datarep: dict,
    gt_datarep: dict,
    skel: Skeleton,
    save_dir: str,
):
    block_nof = clusters.block_nof
    window_slide_nof = int(3 * block_nof / 2) # hyper param 
    window_start_idx = 0 
    _, guide_feat_start_idx, guide_feat_end_idx, _, _ = utils.get_feature_dims(guide_datarep)
    
    # go through files to convert
    datagen_bar = tqdm(range(len(motions)))
    datagen_bar.set_description("Converting every motion files")
    frame_log = tqdm(total=0, position=1, bar_format='{desc}')
    
    window_nof = n_block * block_nof
    for i in datagen_bar:
        motion = motions[i]
        nof = motion.shape[0]
        
        fidx = window_start_idx
        window_gt = []
        window_target = []
        window_guide = []
        while fidx + window_nof + 1 < nof: # +1 to get full disp information 
            frame_log.set_description_str(f"--> Processing frame {fidx}/{nof}")
            end_idx = fidx + window_nof + 1
            window = motion[fidx:end_idx].copy()
            
            # GT
            window_aligned = transform_motion_to(window, np.eye(4))
            gt = motion_to_feat(window_aligned[:-1], gt_datarep, skel)
            
            # get motion
            inputA = window_aligned[:block_nof+1].copy()
            inputB = window_aligned[-(block_nof+1):].copy()

            # window to blocks
            window_blocks = []
            for b in range(n_block):
                start_idx = b * block_nof
                end_idx = start_idx + block_nof + 1 # +1 for full disp info
                block = window_aligned[start_idx:end_idx].copy()
                window_blocks.append(block)
            window_blocks = np.stack(window_blocks, axis=0)
            
            # 1. map to clusters
            window_cidx = clusters.motion_to_cluster(window_blocks)
            window_vq = clusters.blocks_to_vqvae_input(window_blocks)
            window_vq = window_vq.reshape([-1, *window_vq.shape[2:]])
            window_decoded = clusters.index_to_decoded(window_cidx)
            window_decoded = window_decoded.reshape([-1, *window_decoded.shape[2:]])
            
            # 2. skip pathfinding, directly retrieve from decoded codebook
            sA = window_vq[..., :block_nof, :]
            sB = window_vq[..., -block_nof:, :]
            
            guide_inbetween = window_decoded[..., block_nof:-block_nof, :]
            if n_block == 3: # consider directly connected cases for 3 blocks
                if cluster_graph.adj[window_cidx[0]][window_cidx[-1]] != 0:
                    guide_inbetween = interp_guide(sA, sB, guide_feat_start_idx, guide_feat_end_idx)
            guide_not_aligned = np.concatenate([sA, guide_inbetween, sB], axis=-2)
            guide_not_aligned_exp = np.expand_dims(guide_not_aligned, axis=0)
            
            valid_nof = np.array([guide_not_aligned.shape[0]])
            inputA_exp = np.expand_dims(inputA, axis=0)
            inputB_exp = np.expand_dims(inputB, axis=0)
            
            # Guide
            trans0_root_trf = inputA_exp[..., block_nof, 0, :, :]
            guide = preprocess.get_aligned_guide(guide_not_aligned_exp, trans0_root_trf, block_nof, guide_feat_start_idx, guide_feat_end_idx)

            # Target
            dest_root_trf = preprocess.get_dest_from_guide(guide, valid_nof, block_nof, guide_datarep)
            target = preprocess.get_aligned_target(inputA_exp, inputB_exp, dest_root_trf, target_datarep, skel)
            
            # convert and collect
            gt = torch.from_numpy(gt).to(torch.float32)
            guide = torch.from_numpy(guide.squeeze()).to(torch.float32)
            target = torch.from_numpy(target.squeeze()).to(torch.float32)
            window_gt.append(gt)
            window_guide.append(guide)
            window_target.append(target)
            
            # next
            fidx += window_slide_nof
            
        # collect for single mocap
        window_gt = torch.stack(window_gt)
        window_guide = torch.stack(window_guide)
        window_target = torch.stack(window_target)
        
        # export here and start over to avoid using excessive ram
        if i == (len(datagen_bar) - 1):
            print(f"\nData with {n_block} blocks [length, n_feat]:")
            print(f"\t- GT: {list(window_gt.shape[1:])}")
            print(f"\t- Guide: {list(window_guide.shape[1:])}")
            print(f"\t- Target: {list(window_target.shape[1:])}")
            
        torch.save(window_gt, f"{save_dir}/gt/gt_{n_block}_{i}.pt")
        torch.save(window_guide, f"{save_dir}/guide/guide_{n_block}_{i}.pt")
        torch.save(window_target, f"{save_dir}/target/target_{n_block}_{i}.pt")
        
def generate_mg_data(config: argparse.Namespace):
    print("\n[Data generation for MG]")
    utils_n.set_seeds(config.seed)
    
    # setup save dir
    result_dir = f"{config.dataset_dir}/{config.dataset_name}"
    if Path(result_dir).exists(): # rebuild the dataset from scratch if dataset already exists
        shutil.rmtree(result_dir)
    utils.check_create_dir(result_dir)
    
    # load clusters
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    vqvae_dir = config.vqvae_train_config_dir
    vqvae_train_config = utils.load_config(config.vqvae_train_config_dir, config.vqvae_train_config_name)
    clusters = Clusters(vqvae_train_config, config.skel_path, vqvae_dir, device, config.vqvae_ckpt_name)
    config.block_nof = clusters.block_nof
    skel = clusters.skel

    # load graph
    graph_dir = f"{vqvae_dir}/graph"
    cluster_graph = ClusterTransitionGraph(vqvae_dir, graph_dir)

    # load datarep
    target_datarep = utils.load_config(config.datarep_dir, config.target_datarep_name, to_dict=True)
    guide_datarep = utils.load_config(config.datarep_dir, config.guide_datarep_name, to_dict=True)
    gt_datarep = utils.load_config(config.datarep_dir, config.gt_datarep_name, to_dict=True)
    
    # save updated option and datareps
    utils.save_config(config, result_dir, "mg_datagen")
    utils.save_config(target_datarep, result_dir, "target_datarep")
    utils.save_config(guide_datarep, result_dir, "guide_datarep")
    utils.save_config(gt_datarep, result_dir, "gt_datarep")

    # load split motion
    file_split = utils.load_split_filenames(config.file_split_path)
    motion_split = utils.load_motion_from_split(file_split, config.npy_motion_dir)
    output_split_path = {
        "train": f"{result_dir}/train/by_lengths",
        "val": f"{result_dir}/val/by_lengths" 
    } # should be manually deleted after dividing into smaller files
    for i in output_split_path:
        utils.check_create_dir(f"{output_split_path[i]}/gt")
        utils.check_create_dir(f"{output_split_path[i]}/guide")
        utils.check_create_dir(f"{output_split_path[i]}/target")
        
    # make training set
    window_nob = np.arange(3, config.max_window_nob+1) # total length 3 ~ 8 blocks
    for n_block in window_nob:
        print(f"\nGenerating training data with {n_block} blocks")
        motions = motion_split["train"]
        save_dir = output_split_path["train"]
        motions_to_samples(
            motions, 
            clusters,
            cluster_graph,
            n_block, 
            target_datarep,
            guide_datarep,
            gt_datarep, 
            skel, 
            save_dir
        )
        
    # make validation set
    for n_block in window_nob:
        print(f"\nGenerating validation data with {n_block} blocks")
        motions = motion_split["val"]
        save_dir = output_split_path["val"]
        motions_to_samples(
            motions, 
            clusters,
            cluster_graph,
            n_block, 
            target_datarep,
            guide_datarep,
            gt_datarep, 
            skel,
            save_dir,
        )

def divide_mg_data(data_dir_dict, lengths):
    print(f"\nDividing data files for training")
    groups = ["gt", "guide", "target"]
    
    for key, data_dir in data_dir_dict.items():
        # load all data generated per mocap
        data_dir_all = f"{data_dir}/by_lengths"
        print(f"Loading from {data_dir}")
        
        batch_num = 128
        for group in groups:
            load_data_dir = f"{data_dir_all}/{group}"
            out_data_dir = f"{data_dir}/{group}"
            utils.check_create_dir(out_data_dir)
            data_fnames = utils.list_files_in_dir(load_data_dir, False)
            
            # go through each lengths
            for l in lengths:
                prefix = f"{group}_{str(l)}"                
                same_len_data = []
                for f in data_fnames:
                    if f.startswith(prefix):
                        data = torch.load(f"{load_data_dir}/{f}")
                        same_len_data.append(data)
                same_len_data = torch.cat(same_len_data, dim=0)
                    
                n_output_files = (same_len_data.shape[0] + batch_num) // batch_num
                for i in range(n_output_files):
                    start_idx = i * batch_num
                    end_idx = min(start_idx + batch_num, len(same_len_data))
                    batch = same_len_data[start_idx:end_idx].clone()
                    out_data_path = f"{out_data_dir}/{group}_{str(l)}_{i:04d}.pt"
                    torch.save(batch, out_data_path)
     
        print("Files saved at:")
        for g in groups:
            print(f"\t- {data_dir}/{g}")

def compute_statistics(data_dir: str, eps=1e-8):
    '''
    
    Output features will be subset of GT features
    '''
    print(f"\nComputing statistics for dataset: {data_dir}")
    group_names = ["gt", "guide", "target"]
    
    total_count = 0
    total_sum = {}
    total_sq_sum = {}
    statistics = {}
    for i, group in enumerate(group_names): # gt, vq, code
        print(f"\t- Going through {group} files...")
        group_dir = f"{data_dir}/train/{group}"
        data_fnames = utils.list_files_in_dir(group_dir, False)
        
        sample = torch.load(f"{group_dir}/{data_fnames[0]}")
        total_sum = torch.zeros(sample.shape[-1])
        total_sq_sum = torch.zeros(sample.shape[-1])
        for f in data_fnames:
            data = torch.load(f"{group_dir}/{f}")
            data_sum = data.sum(dim=(0, 1))
            data_sq_sum = (data ** 2).sum(dim=(0, 1))
            
            total_sum += data_sum
            total_sq_sum += data_sq_sum

            if i == 0:
                nom = data.shape[0]
                nof = data.shape[1]
                total_count += nom * nof
    
        # statistics[f"{group}_mean"] = 
        mean = total_sum / total_count
        variance = (total_sq_sum / total_count) - (mean ** 2)
        std = torch.sqrt(torch.clamp(variance, min=0.0) + eps)
        statistics[f"{group}_mean"] = mean
        statistics[f"{group}_std"] = std
    
    torch.save(statistics, f"{data_dir}/statistics.pt")
    print(f"Saved statistics: {data_dir}/statistics.pt")

def compute_disc_statistics(config):
    # compute stats for disc input
    print(f"\nComputing statistics for discriminator...")
    stats_name = "statistics_disc"
    result_dir = f"{config.dataset_dir}/{config.dataset_name}"
    # Rebuild the dataset from scratch so files from an older split cannot be reused.
    if Path(result_dir).exists():
        shutil.rmtree(result_dir)

    # load motion
    file_split = utils.load_split_filenames(config.file_split_path)
    motion_split = utils.load_motion_from_split(file_split, config.npy_motion_dir)
    motion_list = motion_split["train"]
    skel = bvh.load(config.skel_path, load_motion=False).skel
    noj = skel.noj()

    # init data format    
    total_count = 0
    eps = 1e-8
    total_sum = np.zeros([noj*3,])
    total_sq_sum = np.zeros([noj*3,])
    statistics = {}
    with tqdm(motion_list, desc="Converting motion files", unit="file") as pbar:
        for i, motion in enumerate(pbar):
            # 1. compute fk to get world position
            world_trf = fk(motion, skel)
            world_pos = world_trf[..., :3, 3]
            
            # 2. compute trans disp for all frames
            world_trans_disp = get_joint_trans_disp(world_pos, pad=False) # [nof-1, noj, 3]
            world_trans_disp = world_trans_disp.reshape([world_trans_disp.shape[0], -1])
            
            # 3. keep sum and squared sum of disp
            trans_disp_sum = world_trans_disp.sum(axis=0)
            trans_disp_sq_sum = (world_trans_disp ** 2).sum(axis=0)
            
            # 4. gather info
            total_sum += trans_disp_sum
            total_sq_sum += trans_disp_sq_sum
            total_count += world_trans_disp.shape[0]
            
        # compute mean and std
        mean = total_sum / total_count
        variance = (total_sq_sum / total_count) - (mean ** 2)
        std = np.sqrt(np.clip(variance, a_min=0.0, a_max=None) + eps)
        
        mean = torch.from_numpy(mean).to(torch.float32)
        std = torch.from_numpy(std).to(torch.float32)
        statistics["mean"] = mean
        statistics["std"] = std
        torch.save(statistics, f"{result_dir}/{stats_name}.pt")
        
def main(config: argparse.Namespace):
    # process motion for paired motion generation dataset
    generate_mg_data(config)
    
    # divide data into small files
    data_dir_dict = {
        "train": f"{config.dataset_dir}/{config.dataset_name}/train",
        "val": f"{config.dataset_dir}/{config.dataset_name}/val"
    }
    lengths = np.arange(3, config.max_window_nob+1)
    divide_mg_data(data_dir_dict, lengths)
    
    if config.stats:
        data_dir = f"{config.dataset_dir}/{config.dataset_name}"
        compute_statistics(data_dir)
        compute_disc_statistics(config)


if __name__ == "__main__":
    '''
    Generate data pairs of [target, guide, GT]
    Target and guide will be concatenated into input later.
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="mg_datagen")
    args = parser.parse_args()
    config = utils.load_config("./nms/config", args.config)
    
    if config.dataset_name is None:
        print("Please provide dataset_name")
        sys.exit()
    
    utils.print_argument(config)
    main(config)
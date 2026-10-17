from typing import Tuple
import torch
import numpy as np
import argparse
import sys
import nms.utils.bvh as bvh
import nms.utils.utils as utils
import nms.utils.network_utils as utils_n
from nms.training.process import motion_to_feat
from nms.motion.motion_op_numpy import transform_motion_to
from tqdm.auto import tqdm
from nms.motion.motion import Skeleton
from typing import List

def motions_to_samples(
    motions: List[np.ndarray],
    block_nof: int, 
    slide_nof: int, 
    datarep: dict,
    skel: Skeleton
) -> Tuple[np.ndarray, np.ndarray]:
    # go through files to convert
    datagen_bar = tqdm(range(len(motions)))
    datagen_bar.set_description("Converting every motion files")
    frame_log = tqdm(total=0, position=1, bar_format='{desc}')
    
    total_processed = []
    total_motion_blocks = []
    for i in datagen_bar:
        motion = motions[i]
        nof = motion.shape[0]
        fidx = 0

        motion_blocks = []
        processed = []
        
        # Fill all disp with valid values
        while fidx + block_nof + 1 < nof: # process block_nof + 1 frames for velocity
            frame_log.set_description_str(f"--> Processing frame {fidx}/{nof}")
            
            # split into blocks & convert to vectors
            block = motion[fidx:fidx+block_nof+1].copy()
            block = transform_motion_to(block, np.eye(4), pivot_idx=0)
            motion_blocks.append(block)
            m = motion_to_feat(block, datarep, skel)
            
            # convert to vectors
            m = m[:-1, ...] # cut last frame
            processed.append(m)
            fidx = fidx + slide_nof
        
        # split train and validation for each motion 
        processed = np.stack(processed, axis=0)
        motion_blocks = np.stack(motion_blocks, axis=0)
        
        # store to collect for all motion data
        total_processed.append(processed)
        total_motion_blocks.append(motion_blocks)
        
    frame_log.close()
    
    total_processed = np.concatenate(total_processed, axis=0)
    total_motion_blocks = np.concatenate(total_motion_blocks, axis=0)
    return total_processed, total_motion_blocks

def generate_vqvae_data(config: argparse.Namespace):
    '''
    Converts 4*4 matrices into blocks of input vectors for VQVAE training
    '''
    print("[VQVAE Data generation]")
    utils_n.set_seeds(config.seed)
    
    # load data represenation
    datarep = utils.load_config(config.datarep_dir, config.datarep_name, True)
    skel = bvh.load(config.skel_path, load_motion=False).skel
    
    # setup save dir
    result_dir = f"{config.dataset_dir}/{config.dataset_name}"
    utils.check_create_dir(result_dir)

    # load split motion
    file_split = utils.load_split_filenames(config.file_split_path) # split file names
    motion_split = utils.load_motion_from_split(file_split, config.npy_motion_dir)
    
    # generate data
    train_files = motion_split["train"]
    val_files = motion_split["val"]
    train_processed, _ = motions_to_samples(train_files, config.block_nof, config.slide_nof, datarep, skel)
    val_processed, val_motion_blocks = motions_to_samples(val_files, config.block_nof, config.slide_nof, datarep, skel)
    print(f" - Train samples shape: {train_processed.shape}")
    print(f" - Validation samples shape: {val_processed.shape}")

    # save generated data
    utils.save_config(config, result_dir, "vqvae_datagen")
    utils.save_config(datarep, result_dir, "vqvae_datarep")
    train_data = torch.from_numpy(train_processed).to(torch.float32)
    val_data = torch.from_numpy(val_processed).to(torch.float32)
    torch.save(train_data, f"{result_dir}/train_data.pt")
    torch.save(val_data, f"{result_dir}/val_data.pt")
    print(f" - Saved at: {result_dir}")
    
def main(config: argparse.Namespace):
    # preprocess motion into blocks of vectors
    generate_vqvae_data(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="vqvae_datagen")
    args = parser.parse_args()
    config = utils.load_config("./nms/config", args.config)
    
    if config.dataset_name is None:
        print("Please provide dataset_name")
        sys.exit()
    
    utils.print_argument(config)
    main(config)


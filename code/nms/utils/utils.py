import torch
import numpy as np
import os
import yaml
import re
import argparse
from typing import Tuple, List, Dict
from pathlib import Path
from tqdm import tqdm
from nms.motion.motion_op_numpy import transform_motion_to

'''
Data
'''
def get_joint_map(original_joints: List, target_joints: List) -> np.ndarray:
    '''
    Get ndarray of integers that indicate the index of each joints in original_joints

    Args:
        original_joints: list of all joints in a skeleton
        target_joints: list of joints to get index
    '''
    noj = len(target_joints)
    joint_map = np.zeros((noj), dtype=np.int32)
    for i in range(noj):
        try:
            jidx = original_joints.index(target_joints[i])
        except ValueError:
            print(f"{target_joints[i]} is not in the original joint list")
        joint_map[i] = jidx
    return joint_map

def dict_to_namespace(config_dict: dict):
    if isinstance(config_dict, dict):
        namespace = argparse.Namespace()
        for key, value in config_dict.items():
            setattr(namespace, key, dict_to_namespace(value))
        return namespace
    elif isinstance(config_dict, list):
        # Convert each element of the list recursively
        return [dict_to_namespace(item) for item in config_dict]
    else:
        return config_dict

def namespace_to_dict(ns):
    result = {}
    for key, value in vars(ns).items():
        if isinstance(value, argparse.Namespace):
            result[key] = namespace_to_dict(value)
        else:
            result[key] = value
    return result

def load_config(dir: str, name: str, to_dict: bool = False):
    config_path = f"{dir}/{name}.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if not to_dict:
        config = dict_to_namespace(config_dict)
    else:
        config = config_dict
    return config

def save_config(config: argparse.Namespace, dir: str, name: str):
    check_create_dir(dir)
    if isinstance(config, argparse.Namespace):
        config_dict = namespace_to_dict(config)
    else:
        config_dict = config

    config_path = f"{dir}/{name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)  # sort_keys=False preserves order

def get_feature_dims(datarep: Dict) -> Tuple:
    # character_config = datarep["character"]
    motion_config = datarep["motion"]

    dim_pointer = 0
    feat_start_idx = {}
    feat_end_idx = {}
    feat_dims = {}
    feat_w = []

    keys = list(motion_config.keys())
    for feat_name in keys:
        config = motion_config[feat_name]
        feat_start_idx[feat_name] = dim_pointer
        noj = len(config["joint"])
        feat_dims[feat_name] = noj * config["dim"]
        dim_pointer += noj * config["dim"]
        feat_end_idx[feat_name] = dim_pointer

        # get weights for each joints
        if config["weight"]:
            weight = np.array(config["weight"]) # [noj,]
            weight = weight[:, np.newaxis] # [noj, 1]
        else:
            weight = np.array([1.0])
            weight = np.repeat(weight, noj, axis=0)
        weight = np.repeat(weight, config["dim"], axis=-1)
        feat_w = np.append(feat_w, weight.flatten())

    total_dim = dim_pointer
    return total_dim, feat_start_idx, feat_end_idx, feat_w, feat_dims

def feat_to_dict(feat, feat_start_idx, feat_end_idx):
    feat_dict = {}
    keys = list(feat_start_idx.keys())
    for k in keys:
        feat_dict[k] = feat[..., feat_start_idx[k]:feat_end_idx[k]]
    return feat_dict

def compute_pairwise_similarity(z_flattened, embedding_weight, distance='l2'):
    """
    Compute L2 or Cosine distance between input vectors (z_flattened) and embeddings.

    Args:
        z_flattened (np.ndarray): Input vectors of shape (B, D).
        embedding_weight (np.ndarray): Codebook vectors of shape (N, D).
        distance_type (str): 'l2' for L2 distance, 'cos' for cosine similarity.

    Returns:
        np.ndarray: Distance matrix of shape (B, N).
    """
    if distance == 'l2':
        # L2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 * (z @ e.T)
        d = - np.sum(z_flattened ** 2, axis=1, keepdims=True) - \
            np.sum(embedding_weight ** 2, axis=1) + \
            2 * np.dot(z_flattened, embedding_weight.T)

    elif distance == 'cos':
        # Normalize input and codebook vectors for cosine similarity
        normed_z_flattened = z_flattened / np.linalg.norm(z_flattened, axis=-1, keepdims=True)
        normed_codebook = embedding_weight / np.linalg.norm(embedding_weight, axis=1, keepdims=True)

        # Compute cosine similarity
        d = np.dot(normed_z_flattened, normed_codebook.T)

    return d

def cos_sim(vec1: np.ndarray, vec2: np.ndarray):
    normed_vec1 = vec1 / (np.linalg.norm(vec1, axis=-1, keepdims=True) + 1e-8) # ! need to check shape after this process
    normed_vec2 = vec2 / (np.linalg.norm(vec2, axis=-1, keepdims=True) + 1e-8)
    cos_sim = np.sum(normed_vec1 * normed_vec2, axis=-1) # cosine similarity
    return cos_sim

def cos_dist(vec1: np.ndarray, vec2: np.ndarray):
    sim = cos_sim(vec1, vec2)
    cos_dist = 1 - sim
    return cos_dist

def get_test_input_blocks(test: argparse.Namespace, block_nof: int):
    '''
    always two inputs
    input_blocks: first input trimmed to last block_nof, second input trimmed to first block_nof
    '''
    input_blocks = []
    for i in range(2):
        cur_input = test.input[i]
        motion_dir = cur_input.motion_dir
        motion_name = cur_input.motion_name
        start_idx = cur_input.start_idx
        end_idx = cur_input.end_idx

        motion = np.load(f"{motion_dir}/{motion_name}.npy")
        input_motion = motion[start_idx:end_idx].copy()

        if i==0:
            block = input_motion[-(block_nof+1):]
        else: # second input
            block = input_motion[:block_nof+1]
        block = transform_motion_to(block, np.eye(4))
        input_blocks.append(block)

    input_blocks = np.stack(input_blocks, axis=0)
    return input_blocks


def get_test_input_blocks_single_frame(test: argparse.Namespace, block_nof: int):
    '''
    always two inputs
    input_blocks: first input trimmed to last block_nof, second input trimmed to first block_nof
    '''
    input_blocks = []
    for i in range(2):
        cur_input = test.input[i]
        motion_dir = cur_input.motion_dir
        motion_name = cur_input.motion_name
        start_idx = cur_input.start_idx
        end_idx = cur_input.end_idx

        motion = np.load(f"{motion_dir}/{motion_name}.npy")
        input_motion = motion[start_idx:end_idx].copy()

        if i==0:
            block = input_motion[-(block_nof+1):]
            pose = block[-1:]
        else: # second input
            block = input_motion[:block_nof+1]
            pose = block[:1]

        # repeat the same pose
        block_single_pose = np.repeat(pose, block_nof+1, axis=0)
        block_single_pose = transform_motion_to(block_single_pose, np.eye(4))
        input_blocks.append(block_single_pose)

    input_blocks = np.stack(input_blocks, axis=0)
    return input_blocks

def get_full_length_test_inputs(tests: argparse.Namespace):
    '''
    inputs: full length input motions
    '''
    inputs = []
    for i in range(len(tests)):
        test = tests[i]
        for j in range(len(test.input)):
            cur_input = test.input[j]
            motion_dir = cur_input.motion_dir
            motion_name = cur_input.motion_name
            start_idx = cur_input.start_idx
            end_idx = cur_input.end_idx
            if i != (len(tests) - 1):
                if j == 0: # first input
                    motion = np.load(f"{motion_dir}/{motion_name}.npy")
                    input_motion = motion[start_idx:end_idx].copy()
                    input_motion = transform_motion_to(input_motion, np.eye(4))
                    inputs.append(input_motion)
            else: # last test sample
                motion = np.load(f"{motion_dir}/{motion_name}.npy")
                input_motion = motion[start_idx:end_idx].copy()
                input_motion = transform_motion_to(input_motion, np.eye(4))
                inputs.append(input_motion)
    return inputs


'''
File IO
'''
def check_create_dir(dir: str) -> None:
    Path(dir).mkdir(parents=True, exist_ok=True)

def list_files_in_dir(file_dir: str, numerical: bool = True) -> List[str]:
    '''
    List all files in dir.

    Args:
        file_dir: directory to retreive file names from
    '''

    files = os.listdir(file_dir)
    files = [f for f in files if os.path.isfile(f"{file_dir}/{f}")]
    if numerical:
        sorted_filenames = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
    else:
        files.sort()
        sorted_filenames = files
    return sorted_filenames


def load_motion_from_split(file_split: dict, motion_dir: str) -> dict:
    '''

    Return:
        motion_split = dict[str, list[np.ndarray]]
    '''
    motion_split = {}
    for key, filenames in file_split.items():
        motion_list = []
        with tqdm(filenames, desc=f"Importing {key} motion files", unit="file") as pbar:
            for filename in pbar:
        # for i in filenames:
                file_path = f"{motion_dir}/{filename}"
                motion_npy = np.load(file_path)
                motion_list.append(motion_npy)
        motion_split[key] = motion_list
    return motion_split


def load_split_filenames(path: str) -> dict:
    '''

    Return:
        file_split = dict[str, list[str]]
    '''
    with open(path, "r") as f:
        file_split = yaml.load(f, Loader=yaml.FullLoader)
    return file_split

def is_pt(fname: str) -> bool:
    if fname[-3:] == ".pt":
        return True
    return False


def is_bvh(fname: str) -> bool:
    if fname[-4:] == ".bvh":
        return True
    return False


'''
Logging, plotting
'''

def print_argument(opt: argparse.Namespace):
    print('------------ Options -------------')
    for k, v in vars(opt).items():
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    return opt

def print_model(model: torch.nn.Module):
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

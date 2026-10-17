import numpy as np
import sys
import argparse
import shutil
import torch
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix, save_npz
import nms.utils.utils as utils
from nms.modules.cluster import Clusters
from nms.modules.graph import ClusterTransitionGraph
    
def convert_motion_to_cidx(
    clusters: Clusters,
    motion: np.ndarray, 
    init_fidx: int, 
    buffer_size: int = 64
) -> np.ndarray:
    fidx = init_fidx
    block_nof = clusters.block_nof
    
    nof = motion.shape[0]
    sample_buffer = []
    cidx_seq = []
    frame_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
    while fidx + block_nof + 1 < nof:
        frame_log.set_description_str(f"--> Processing frame {fidx}/{nof}")
        end_idx = fidx + block_nof + 1
        
        block = motion[fidx:end_idx].copy()
        sample_buffer.append(block)
        
        # check buffer. if full, forward vqvae and append index seq 
        if len(sample_buffer) == buffer_size:
            block_batch = np.stack(sample_buffer, axis=0)
            cluster_indices = clusters.motion_to_cluster(block_batch)

            sample_buffer.clear()
            cidx_seq.append(cluster_indices)
        fidx += block_nof
    frame_log.close()
    
    # encode remaining blocks in the buffer
    if len(sample_buffer) != 0:
        block_batch = np.stack(sample_buffer, axis=0)
        cluster_indices = clusters.motion_to_cluster(block_batch)
        cidx_seq.append(cluster_indices)
            
    cidx_seq = np.concatenate(cidx_seq, axis=0)
    return cidx_seq
            
def generate_cidx_seq(
    clusters: Clusters,
    motions: list,
    filenames: list,
    result_dir: str
):
    block_nof = clusters.block_nof
    slide_nof = block_nof // 2
    
    # generate samples
    cidx_seq_list = []
    with tqdm(motions, desc="Converting motion files", unit="file") as pbar:
        for i, motion in enumerate(pbar):
            first_init_fidx = 0
            first_cidx_seq = convert_motion_to_cidx(clusters, motion, first_init_fidx)
            cidx_seq_list.append(first_cidx_seq)
            
            second_init_fidx = slide_nof
            second_cidx_seq = convert_motion_to_cidx(clusters, motion, second_init_fidx)
            cidx_seq_list.append(second_cidx_seq)
            
    # save each sequences in separate files
    assert len(cidx_seq_list) == 2 * len(filenames), "Number of created cluster index sequence does not match with motion list"
    for i in range(len(filenames)):
        motion_name = filenames[i]
        np.save(f"{result_dir}/{motion_name[:-4]}_1.npy", cidx_seq_list[2*i])
        np.save(f"{result_dir}/{motion_name[:-4]}_2.npy", cidx_seq_list[2*i+1])
    print(f"Total: {len(cidx_seq_list)} sequences generated")

def extract_cidx_valid_trans(codebook_size, result_dir):
    valid_trans_count = np.zeros((codebook_size, codebook_size), dtype=np.int32)

    # keep track of cidx transition
    cidx_seq_fnames = utils.list_files_in_dir(f"{result_dir}/seq", False)
    for fname in cidx_seq_fnames:
        cidx_seq = np.load(f"{result_dir}/seq/{fname}")
        cur_cidx = cidx_seq[:-1]
        next_cidx = cidx_seq[1:]
        np.add.at(valid_trans_count, (cur_cidx, next_cidx), 1) # increment if there is a transition
    valid_trans_bool = (valid_trans_count > 0).astype(int)
    return valid_trans_bool

def create_valid_node_filter(valid_trans_bool, codebook):
    n_components, labels = connected_components(csgraph=csr_matrix(valid_trans_bool), directed=True, connection="strong")
    components = [np.where(labels == k)[0] for k in range(n_components)]
    components.sort(key=len, reverse=True)
    for k, nodes in enumerate(components):
        print(f"component {k}: {nodes}")

    # find valid nodes(== clusters)
    largest_comp_idx = np.bincount(labels).argmax()
    largest_comp_nodes = np.where(labels == largest_comp_idx)[0]
    not_in_largest = np.where(labels != largest_comp_idx)[0]

    # map invalid(not connected) nodes to largest component
    valid_node_filter = np.arange(valid_trans_bool.shape[0])
    valid_codes = codebook[largest_comp_nodes]
    invalid_codes = codebook[not_in_largest]
    sim = utils.compute_pairwise_similarity(invalid_codes, valid_codes, "cos")
    closest_code_idx = np.argmax(sim, axis=1)
    valid_node_filter[not_in_largest] = largest_comp_nodes[closest_code_idx]
    return valid_node_filter

def check_connectivity(ctg: ClusterTransitionGraph):
    ##### check whatever I want with constructed graph
    adj_matrix = ctg.adj
    
    ### 1. Strongly connected?
    # convert to sparse representation for efficiency
    csr = csr_matrix(adj_matrix)

    # compute connected components
    n_components, labels = connected_components(csgraph=csr, directed=True, connection='strong')

    # check if the graph is fully connected (i.e., only 1 component)
    if n_components == 1:
        print("\nAll nodes are connected.")
    else:
        print(f"\nThe graph is NOT fully connected. It has {n_components} components.")

    ### 2. Self connection in directed graph?
    diag = adj_matrix.diagonal()
    diag_nonzero = diag.sum()
    print(f"Number of self-connected nodes: {diag_nonzero}")

def main(config: argparse.Namespace):
    # load cluster
    vqvae_dir = config.vqvae_train_config_dir
    vqvae_train_config = utils.load_config(config.vqvae_train_config_dir, config.vqvae_train_config_name)
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    clusters = Clusters(vqvae_train_config, config.skel_path, vqvae_dir, device, config.load_ckpt_name, load_valid_node_filter=False)

    # make graph dir
    graph_dir = f"{vqvae_dir}/graph"
    utils.check_create_dir(graph_dir)

    # compute adjacency in mocap data
    file_split_path = config.file_split_path
    npy_motion_dir = config.npy_motion_dir

    file_split = utils.load_split_filenames(file_split_path)
    motion_split = utils.load_motion_from_split(file_split, npy_motion_dir)
    
    # map all motion blocks to clusters, get sequence with cluster indices
    filenames = file_split["train"]
    motions = motion_split["train"]
    cidx_seq_dir = f"{graph_dir}/seq"
    utils.check_create_dir(cidx_seq_dir)
    generate_cidx_seq(clusters, motions, filenames, cidx_seq_dir)
    
    # keep track of valid transition between clusters
    codebook_size = clusters.codebook_size
    valid_trans_bool = extract_cidx_valid_trans(codebook_size, graph_dir) # adjacency for cluster graph
    shutil.rmtree(cidx_seq_dir)
    save_npz(f"{graph_dir}/valid_trans_bool.npz", csr_matrix(valid_trans_bool))
    print("Saved valid_trans_bool!")
    
    valid_node_filter = create_valid_node_filter(valid_trans_bool, clusters.codebook)
    np.save(f"{graph_dir}/valid_node_filter.npy", valid_node_filter)
    print("Saved valid_node_filter!")

    if config.precompute:
        weights = { 
            "root_height": config.w_root_height,
            "root_orient": config.w_root_orient,
            "root_trans_disp": config.w_root_trans_disp,
            "root_ang_disp": config.w_root_ang_disp,
            "joint_pos_root_rel": config.w_joint_pos_root_rel,
            "joint_trans_disp": config.w_joint_trans_disp,
            "code": config.w_code,
        }
        graph = ClusterTransitionGraph(vqvae_dir, graph_dir, weights=weights, load_dist_cache=False)
        graph.precompute_all_pair_cost()
        graph.save_cost(graph_dir)
    else:
        # build graph based on valid_trans_bool
        graph = ClusterTransitionGraph(vqvae_dir, graph_dir, load_dist_cache=False)
    check_connectivity(graph)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="construct_graph")
    args = parser.parse_args()
    config = utils.load_config("./nms/config", args.config)
    if config.vqvae_train_config_dir == "" or config.vqvae_train_config_name == "":
        print("Please provide vqvae train config dir and name")
        sys.exit()
    
    utils.print_argument(config)
    main(config)
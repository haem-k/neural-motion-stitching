import torch
import nms.utils.utils as utils
import nms.utils.network_utils as utils_n
import numpy as np
import nms.utils.bvh as bvh
from nms.training.process import feat_to_root_pos, motion_to_feat, feat_to_world_pos
from nms.motion.motion_op_numpy import fk, transform_motion_to_batch
from nms.modules.cvqvae import CVQVAE
from pathlib import Path

class Clusters:
    def __init__(self, vqvae_train_config, skel_path, vqvae_dir, device, load_ckpt_name="", load_valid_node_filter=True):
        self.device = device
        self.vqvae_dir = vqvae_dir
        self.block_nof = vqvae_train_config.input_len
        self.k_candidate = vqvae_train_config.k_candidate

        self.load_vqvae(vqvae_train_config, load_ckpt_name)        
        self.load_codebook()        
        self.skel = bvh.load(skel_path, load_motion=False).skel

        filter_path = Path(f"{vqvae_dir}/graph/valid_node_filter.npy")
        if not load_valid_node_filter:
            self.valid_node_filter = np.arange(self.codebook_size)
            print("Valid node filter disabled")
        elif filter_path.exists():
            self.valid_node_filter = np.load(filter_path)
            print("Loaded valid_node_filter from file")
        else:
            self.valid_node_filter = np.arange(self.codebook_size)
            print("No valid_node_filter file, initialized identity mapping")
        
        shoulders = ["LeftShoulder", "RightShoulder"] # ????
        self.shoulder_jnt_map = utils.get_joint_map(self.datarep["motion"]["joint_pos_root_rel"]["joint"], shoulders)
        
    def load_vqvae(self, vqvae_train_config, load_ckpt_name):
        # load data representation
        datarep_name = vqvae_train_config.datarep_name
        self.datarep = utils.load_config(self.vqvae_dir, datarep_name, to_dict=True)
        self.total_dim, self.feat_start_idx, self.feat_end_idx, _, _ = utils.get_feature_dims(self.datarep)
        
        # load pretrained vqvae
        vqvae = CVQVAE(vqvae_train_config.archi)
        vqvae, ckpt_name = utils_n.load_model(vqvae, self.vqvae_dir, load_ckpt_name)
        print(f"Loaded checkpoint: {vqvae_train_config.exp_name} ({ckpt_name})")
        
        self.vqvae = vqvae.to(self.device)
        self.vqvae.eval()
        print(f"Prepared vqvae: {vqvae_train_config.exp_name}")

    def load_codebook(self):        
        codebook_dir = f"{self.vqvae_dir}/codebook"
        self.codebook_decoded = np.load(f"{codebook_dir}/codebook_decoded.npy")
        self.codebook = np.load(f"{codebook_dir}/codebook.npy")
        self.codebook_size = self.codebook.shape[0]
    
    def get_n_cluster(self):
        return np.unique(self.valid_node_filter).size
    
    def measure_seq_dist(
        self,
        block_feat: np.ndarray,
        decoded: np.ndarray
    ):
        '''
        
        block_feat: [batch, block_nof, n_feat]
        decoded: [batch, k, block_nof, n_feat]
        '''
        # root pos
        decoded_topk_root_pos = feat_to_root_pos(decoded, self.feat_start_idx, self.feat_end_idx).squeeze()
        block_root_pos = feat_to_root_pos(block_feat, self.feat_start_idx, self.feat_end_idx).squeeze()
        block_root_pos = np.expand_dims(block_root_pos, axis=-3)
        diff_root_pos = decoded_topk_root_pos - block_root_pos
        dist_root_pos = np.linalg.norm(diff_root_pos, axis=-1)
        dist_root_pos = np.mean(dist_root_pos, axis=-1)
        
        dist = dist_root_pos
        return dist
    
    def encode_w_vqvae(
        self,
        vqvae_input: np.ndarray
    ):
        '''
        vqvae_input: [batch, block_nof, n_feat]
        '''
        vqvae_input = torch.from_numpy(vqvae_input).to(torch.float32).to(self.device)
        x_enc, x_q, encoding_indices = self.vqvae.encode(vqvae_input)
        
        x_enc = x_enc.cpu().numpy()
        x_q = x_q.cpu().numpy()
        encoding_indices = encoding_indices.cpu().numpy()
        return x_enc, x_q, encoding_indices
    
    def blocks_to_vqvae_input(
        self,
        blocks: np.ndarray
    ):
        batch_size = blocks.shape[0]
        target_trf = np.expand_dims(np.eye(4), axis=0)
        target_trf = np.repeat(target_trf, batch_size, axis=0)
        blocks = transform_motion_to_batch(blocks.copy(), target_trf)
        
        blocks_feat = motion_to_feat(blocks, self.datarep, self.skel)
        blocks_feat = blocks_feat[..., :-1, :]
        return blocks_feat
    
    '''
    Main function that maps motion blocks to each cluster
    '''
    def motion_to_cluster(
        self,
        blocks: np.ndarray
    ):
        '''
        Returns index of clusters for given motion blocks
        
        Args:
            blocks: motion block to be mapped to clusters [batch, block_nof+1, noj, 4, 4]
            
        Returns:
            cidx: [batch,]
        '''
        assert blocks.ndim == 5, "Check shape of block"
        
        blocks_feat = self.blocks_to_vqvae_input(blocks)
        x_enc, x_q, encoding_indices = self.encode_w_vqvae(blocks_feat) 
        if self.k_candidate == 1:
            return self.valid_node_filter[encoding_indices]
        
        # find top k codes closest to encoded vector
        sim = utils.compute_pairwise_similarity(x_enc, self.codebook, 'cos') # ! [batch, codebook_size] --> 이 pairwise similarity 부터 빼고 시작돼야해
        sim_topk_indices = np.argpartition(-sim, kth=self.k_candidate-1, axis=1)[:, :self.k_candidate] # unsorted, only top k [128, k]
        
        # among the top k cluster, choose the one with closest root in the decoded sequence
        decoded_topk = self.codebook_decoded[sim_topk_indices]
        dist = self.measure_seq_dist(blocks_feat, decoded_topk) # compare only root
        dist_min_indices = np.argmin(dist, axis=-1) 
        
        # get cluster indices
        batch_size = blocks.shape[0]
        batch_indices = np.arange(batch_size)
        cluster_indices = sim_topk_indices[batch_indices, dist_min_indices]
        
        # filter indices
        cluster_indices = self.valid_node_filter[cluster_indices]
        return cluster_indices
    
    def index_to_decoded(
        self,
        cluster_index: np.ndarray
    ):
        decoded = self.codebook_decoded[cluster_index]
        return decoded
    
    def index_to_code(
        self,
        cluster_index: np.ndarray
    ):
        cluster_codes = self.codebook[cluster_index]
        return cluster_codes
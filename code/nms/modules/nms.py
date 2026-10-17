import torch
import argparse
import nms.utils.utils as utils
import nms.utils.network_utils as utils_n
from nms.rotation.rotation_numpy import rotmat_to_euler
import numpy as np
from nms.motion.motion_op_numpy import transform_motion_to, get_parent_rel_pos_from_skel, get_basis_from_forward_vec
from nms.training.process import interp_guide, export_pos_feature_vis_np, motion_to_feat
from nms.modules.cluster import Clusters
from nms.modules.graph import ClusterTransitionGraph
from nms.modules.motion_generator import MotionGenerator
from nms.modules.ib import inertial_blending
import nms.modules.preprocess as preprocess
import nms.utils.bvh as bvh
from nms.motion.motion import Motion
from tqdm import tqdm

class NeuralMotionStitching:
    def __init__(
        self,
        config: argparse.Namespace,
        vqvae_train_config: argparse.Namespace,
        mg_train_config: argparse.Namespace,
        device: torch.device
    ):
        self.vqvae_train_config = vqvae_train_config
        self.mg_train_config = mg_train_config
        self.device = device
        
        # init clusters
        vqvae_dir = config.vqvae_train_config_dir
        self.clusters = Clusters(self.vqvae_train_config, config.skel_path, vqvae_dir, device, config.vqvae_ckpt_name)
        print()
        
        # init graph
        graph_dir = f"{vqvae_dir}/graph"
        weights = {
            "root_height": config.w_root_height,
            "root_orient": config.w_root_orient,
            "root_trans_disp": config.w_root_trans_disp,
            "root_ang_disp": config.w_root_ang_disp,
            "joint_pos_root_rel": config.w_joint_pos_root_rel,
            "joint_trans_disp": config.w_joint_trans_disp,
            "code": config.w_code,
            "control": config.w_control
        }
        self.cluster_graph = ClusterTransitionGraph(vqvae_dir, graph_dir, weights=weights, load_precomputed=config.load_precomputed, control=config.control)
        self.cluster_graph.init_graph() # precompute edge costs
        print()
        
        # init mg
        mg_dir = config.mg_train_config_dir
        mogen = MotionGenerator(self.mg_train_config, device, mode="val")
        mogen.transformer, ckpt_name = utils_n.load_model(mogen.transformer, mg_dir, config.mg_ckpt_name)
        print(f"Loaded checkpoint: {mg_train_config.exp_name} ({ckpt_name})")
        
        mogen.transformer.eval()
        self.mogen = mogen
        print(f"Prepared mg: {mg_train_config.exp_name}")
        
        # init other info
        self.skel = self.mogen.skel
        self.parent_rel_joint_pos = get_parent_rel_pos_from_skel(self.skel)[1:]
        self.max_len = mogen.max_len
        self.block_nof = mogen.block_nof
        self.dummy_trf = np.broadcast_to(np.eye(4), (self.block_nof, self.skel.noj(), 4, 4)).copy() # for padding
        self.dummy_guide = motion_to_feat(self.dummy_trf, self.clusters.datarep, self.skel)
    
    def get_control_input(self, test):
        control = test.input[-1].control
        control_vec = np.array([control[0], 0, control[2]])
        control_vec = control_vec / np.linalg.norm(control_vec)
        return control_vec
    
    def combine_results(
        self,
        inputs, # full length input
        results, # inputA + transition + inputB
        result_dir, # dir to save
        save_bvh=False,
    ):
        input0 = inputs[0]
        combined = [input0]
        dest_root_trf = input0[-(self.block_nof+1), 0] # block_nof + 1 is used for inference
        
        motion_accum = 0
        for i, trans in enumerate(results):
            trans_aligned = transform_motion_to(trans, dest_root_trf, closest=False)
            dest_root_trf = trans_aligned[-self.block_nof, 0]
            
            prev_motion = combined.pop()
            next_input_motion = inputs[i+1].copy()
            next_input_motion_aligned = transform_motion_to(next_input_motion, dest_root_trf, closest=False) ## 여기가 문제인듯? 이러면서 에러 쌓인다
            dest_root_trf = next_input_motion_aligned[-(self.block_nof+1), 0]
            
            combined.append(prev_motion[:-(self.block_nof+1)]) # first motion
            combined.append(trans_aligned[:-self.block_nof]) # transition
            combined.append(next_input_motion_aligned) # second motion
            
            range0 = motion_accum
            rangeE = motion_accum + prev_motion.shape[0] - 1
            print(f"{i}th input range: [{range0}, {rangeE}), {rangeE - range0} frames")
            motion_accum = motion_accum + prev_motion.shape[0] - 1 + trans_aligned.shape[0] - 2 * self.block_nof
        
        range0 = motion_accum
        rangeE = motion_accum + next_input_motion_aligned.shape[0] - 1
        print(f"last input range: [{range0}, {rangeE}), {rangeE - range0} frames")
        
        combined = np.concatenate(combined, axis=0)
        np.save(f"{result_dir}/output_combined.npy", combined)
        if save_bvh:
            bvh.save(Motion.from_matrix(combined, self.skel), f"{result_dir}/output_combined.bvh")
          
    def forward_mogen(
        self,
        inputA: np.ndarray, # in batches
        inputB: np.ndarray, # not aligned to I
        guide_not_aligned: np.ndarray, # in batches, padded to maximum seq in that batch
        valid_nof: np.ndarray, # in batches
    ):
        batch_size = inputA.shape[0]
        max_len = guide_not_aligned.shape[-2]
        
        # Guide
        trans0_root_trf = inputA[..., self.block_nof, 0, :, :]
        guide = preprocess.get_aligned_guide(guide_not_aligned, trans0_root_trf, self.block_nof, self.clusters.feat_start_idx, self.clusters.feat_end_idx)

        # Target
        dest_root_trf = preprocess.get_dest_from_guide(guide, valid_nof, self.block_nof, self.clusters.datarep)
        target = preprocess.get_aligned_target(inputA, inputB, dest_root_trf, self.mogen.target_datarep, self.skel)
        target = preprocess.pad_target_inbetween(target, max_len, self.block_nof, valid_nof)
        
        # Input
        input = np.concatenate([target, guide], axis=-1)

        input = torch.from_numpy(input).to(torch.float32).to(self.device)
        valid_nof = torch.from_numpy(valid_nof).to(torch.int32)
        padding_mask = utils_n.get_square_padding_mask(max_len, valid_nof, self.mogen.transformer.n_head, 1, self.device)
        
        # forward through motion generator
        output = self.mogen.transformer(input, padding_mask)

        # align output
        padding_mask = ~padding_mask[..., 0, 0, :]
        valid_nof = valid_nof.to(self.device)
        local_trfs = self.mogen.get_aligned_local_trfs(input, output, valid_nof, padding_mask=padding_mask)
        targetB_trf = local_trfs["targetB"]
        pred_trf = local_trfs["pred"]
        
        # concat targetB to pred
        valid_nof = valid_nof.detach().cpu().numpy()
        start_idx = np.expand_dims(valid_nof-self.block_nof, axis=-1)
        offset_indices = np.arange(self.block_nof)
        range_indices = start_idx + offset_indices
        batch_indices = np.arange(batch_size).reshape(-1, 1)
        batch_indices = np.broadcast_to(batch_indices, range_indices.shape).copy() # [batch_size, block_nof]
        batch_indices = torch.from_numpy(batch_indices).to(torch.int32)
        range_indices = torch.from_numpy(range_indices).to(torch.int32)
        pred_trf[batch_indices, range_indices] = targetB_trf
        
        return target, guide, pred_trf


    def generate(self, test_input, result_dir, combine=False, control=False, blend=False, single_frame=False, save_bvh=False):
        '''
        Generate motions from test inputs and export the results.
        '''
        n_test = len(test_input.tests)
        
        paths = []
        results = []
        controls = []
        for i in tqdm(range(n_test), desc="Generating from test samples"):
            test = test_input.tests[i]
            if single_frame:
                input_blocks = utils.get_test_input_blocks_single_frame(test, self.block_nof)
            else:
                input_blocks = utils.get_test_input_blocks(test, self.block_nof)
            
            control_input = None
            if control:
                control_input = self.get_control_input(test)
                controls.append(control_input)

            motion, path, guide_trim = self.stitch_motion(
                input_blocks,
                control=control,
                control_input=control_input,
            )
            paths.append(path)
            
            if blend:
                motion_blended = inertial_blending(input_blocks[0], motion[self.block_nof:], 10)
                motion = motion_blended
            results.append(motion)
            
            # save results
            inputA_filename = f"input{i}_A"
            inputB_filename = f"input{i}_B"
            np.save(f"{result_dir}/{inputA_filename}.npy", input_blocks[0])
            np.save(f"{result_dir}/{inputB_filename}.npy", input_blocks[1])
            
            output_filename = f"output{i}"
            np.save(f"{result_dir}/{output_filename}.npy", motion)
            if save_bvh:
                bvh.save(Motion.from_matrix(motion, self.skel), f"{result_dir}/{output_filename}.bvh")
            export_pos_feature_vis_np(guide_trim, self.clusters.datarep, result_dir, f"guide{i}")
            
        # save all paths in text
        with open(f"{result_dir}/result_paths.txt", "w") as f:
            for i in paths:
                f.write(" ".join(map(str, i)) + "\n")  # Convert list to space-separated string
        
        # save control input for visualization
        if control:
            controls = np.stack(controls, axis=0)
            controls_mat = get_basis_from_forward_vec(controls)
            controls_euler = rotmat_to_euler(controls_mat, degrees=True)
            np.save(f"{result_dir}/control_euler.npy", controls_euler)
        print(f"Exported results at: {result_dir}")    

        if combine:
            print("Combining all results...")
            inputs = utils.get_full_length_test_inputs(test_input.tests)
            self.combine_results(inputs, results, result_dir, save_bvh=save_bvh)

    
    def stitch_motion(
        self, 
        input_trfs, # [2, block_nof, noj, 4, 4]
        control=False, 
        control_input=None, # control vector for this sample [3]
    ):
        '''
        Stitch two motion blocks and return the motion, graph path, and guide.
        
        '''
        if control:
            assert control_input is not None, "Control inputs are not given for control NMS"
                
        # 1. map to clusters
        inputs_cidx = self.clusters.motion_to_cluster(input_trfs)
        
        # 2. cluster path finding - A*
        start = inputs_cidx[0]
        goal = inputs_cidx[1]
        if control:
            init_root_trf = input_trfs[0, -2, 0] # use last frame
            path, _, _ = self.cluster_graph.astar_control(start, goal, control_input, init_root_trf)
        else:
            path, _, _ = self.cluster_graph.astar(start, goal)
        path_len = len(path)
        
        # 3. preprocess and forward mogen
        # input motions to vqvae input
        inputs_vq = self.clusters.blocks_to_vqvae_input(input_trfs) # [2, 16, 64]
        decoded_path = self.clusters.index_to_decoded(path) # [3, 16, 64]
        decoded_inbetween = decoded_path[1:-1] # [1, 16, 64]
        decoded_inbetween = np.reshape(decoded_inbetween, [-1, *decoded_inbetween.shape[2:]]) # reshapte to [nof, ndim]
        sA = inputs_vq[0]
        sB = inputs_vq[1]
        
        # consider directly connected cases
        if path_len <= 2:
            decoded_inbetween = interp_guide(sA, sB, self.clusters.feat_start_idx, self.clusters.feat_end_idx)
        guide_not_aligned = np.concatenate([sA, decoded_inbetween, sB], axis=0)
        guide_not_aligned_exp = np.expand_dims(guide_not_aligned, axis=0)
        
        # prepare valid_nof
        valid_nof = np.array([guide_not_aligned.shape[-2]])
        inputA_exp = input_trfs[0:1]
        inputB_exp = input_trfs[1:2]
        
        # infer transformer
        target, guide, pred_trf = self.forward_mogen(inputA_exp, inputB_exp, guide_not_aligned_exp, valid_nof)
        
        # export motion and guide
        nof = valid_nof[0]
        guide_trim = guide[:, :nof] # export function requires this to be in batch
        motion = pred_trf[0, :nof, ...].detach().cpu().numpy()
        motion[..., 1:, :3, 3] = self.parent_rel_joint_pos
        return motion, path, guide_trim
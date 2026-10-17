import numpy as np
import nms.utils.utils as utils
from nms.motion.motion import Skeleton
from nms.rotation.rotation_numpy import rotmat_pos_to_trf, rotmat_to_quat, ortho6d_to_rotmat, rotmat_to_ortho6d, aaxis_to_rotmat, quat_closest
from nms.motion.motion_op_numpy import transform_motion_to_batch
from nms.training.process import get_last_frame_with_offset_np, recon_orient_from_disp, recon_pos_from_disp, motion_to_feat

def get_aligned_guide(
    guide_not_aligned: np.ndarray,
    trans0_root_trf: np.ndarray,
    block_nof: int,
    guide_feat_start_idx: dict, # assume that vq datarep is the same as guide datarep
    guide_feat_end_idx: dict
):
    '''
    process in batches

    guide_not_aligned: [batch_size, max_len, guide_total_dim]
    start_root_trf: first root transformation for predicted frames [4, 4]
    '''
    # guide will already be padded, so its length equals maximum nof for this batch
    max_path_len = guide_not_aligned.shape[-2] // block_nof # ! need to check this when I batch generate

    # should mask invalid blocks
    up_axis = np.array([0.0, 1.0, 0.0])
    start_root_orient = trans0_root_trf[..., :3, :3]
    guide = guide_not_aligned[..., :block_nof, :].copy()
    block_queue = guide_not_aligned[..., block_nof:, :].copy()
    for i in range(max_path_len - 1):
        start_fidx = i * block_nof
        end_fidx = start_fidx + block_nof

        block = block_queue[..., start_fidx:end_fidx, :]
        block_guide = block.copy()

        root_orient = block[..., guide_feat_start_idx["root_orient"]:guide_feat_end_idx["root_orient"]]
        root_orient = ortho6d_to_rotmat(root_orient)
        anchor = root_orient[..., :1, :, :] # original first root orient as an anchor
        anchor_inv = np.linalg.inv(anchor)

        # find closest orient destination with start_root_orient as quat_target
        anchor_quat = rotmat_to_quat(anchor).squeeze()
        start_root_orient_quat = rotmat_to_quat(start_root_orient).squeeze()
        _, theta = quat_closest(start_root_orient_quat, anchor_quat, up_axis)
        theta = np.expand_dims(theta, axis=-1)
        up_axis_repeat = np.repeat(up_axis[None, :], theta.shape[0], axis=0)
        dR_aaxis = theta * up_axis_repeat

        dR = aaxis_to_rotmat(dR_aaxis)
        dR = np.expand_dims(dR, axis=-3)
        root_orient_dest = dR @ anchor

        # align root orient
        root_orient_anchor = anchor_inv @ root_orient
        root_orient_rotated = root_orient_dest @ root_orient_anchor
        root_orient_rotated_6d = rotmat_to_ortho6d(root_orient_rotated)
        block_guide[..., guide_feat_start_idx["root_orient"]:guide_feat_end_idx["root_orient"]] = root_orient_rotated_6d
        start_root_orient = root_orient_rotated[..., -1, :, :] # update start_root_orient

        # align translational displacement
        root_trans_disp = block[..., guide_feat_start_idx["root_trans_disp"]:guide_feat_end_idx["root_trans_disp"]]
        root_trans_disp = np.expand_dims(root_trans_disp, axis=-2)
        root_trans_disp = np.swapaxes(root_trans_disp, -1, -2)
        root_trans_disp_anchor = anchor_inv @ root_trans_disp # need to convert reference frame first
        root_trans_disp_rotated = root_orient_dest @ root_trans_disp_anchor
        root_trans_disp_rotated = np.swapaxes(root_trans_disp_rotated, -1, -2)
        block_guide[..., guide_feat_start_idx["root_trans_disp"]:guide_feat_end_idx["root_trans_disp"]] = root_trans_disp_rotated.squeeze()

        # align joint trans disp
        joint_trans_disp = block[..., guide_feat_start_idx["joint_trans_disp"]:guide_feat_end_idx["joint_trans_disp"]] # [block_nof, feat noj]
        joint_trans_disp = np.reshape(joint_trans_disp, [*joint_trans_disp.shape[:-1], -1, 3])
        joint_trans_disp = np.swapaxes(joint_trans_disp, -1, -2)
        joint_trans_disp_anchor = anchor_inv @ joint_trans_disp
        joint_trans_disp_rotated = root_orient_dest @ joint_trans_disp_anchor
        joint_trans_disp_rotated = np.swapaxes(joint_trans_disp_rotated, -1, -2)
        joint_trans_disp_rotated = np.reshape(joint_trans_disp_rotated, [*joint_trans_disp_rotated.shape[:-2], -1])
        block_guide[..., guide_feat_start_idx["joint_trans_disp"]:guide_feat_end_idx["joint_trans_disp"]] = joint_trans_disp_rotated

        # update guide
        guide = np.concatenate([guide, block_guide], axis=-2)
    return guide

def get_dest_from_guide(guide: np.ndarray, valid_nof: np.ndarray, block_nof: int, datarep: dict):
    '''
    Compute next target root from guide
    Orientation is computed from the last block with angular disp
    Position is computed by accumulating from the start of the guide

    Args:
        guide: [..., infer_nof, n_feat]
        datarep: guide data representation

    Returns:
        root_trf: [..., 4, 4]

    '''
    _, feat_start_idx, feat_end_idx, _, _= utils.get_feature_dims(datarep)

    # orientation - add displacement at the last orientation
    guide_last = get_last_frame_with_offset_np(guide, valid_nof, block_nof+1)
    guide_last = np.expand_dims(guide_last, axis=-2)

    last_root_orient = guide_last[..., -1:, feat_start_idx["root_orient"]:feat_end_idx["root_orient"]]
    last_root_orient = ortho6d_to_rotmat(last_root_orient)
    root_ang_disp = guide_last[..., -1:, feat_start_idx["root_ang_disp"]:feat_end_idx["root_ang_disp"]]
    root_ang_disp = ortho6d_to_rotmat(root_ang_disp)
    root_orient = recon_orient_from_disp(root_ang_disp[:, None, :], last_root_orient[:, None, :])
    root_orient = root_orient[..., -1, 0, :, :]

    # position - should be reconstructed from the beginning
    init_root_h = guide[..., :1, feat_start_idx["root_height"]:feat_end_idx["root_height"]]
    zero = np.zeros_like(init_root_h)
    init_root_pos = np.concatenate([zero, init_root_h, zero], axis=-1)
    root_trans_disp = guide[..., feat_start_idx["root_trans_disp"]:feat_end_idx["root_trans_disp"]]
    root_trans_disp = np.expand_dims(root_trans_disp, axis=-2)
    root_pos = recon_pos_from_disp(root_trans_disp, init_root_pos[:, None, :])
    root_pos = get_last_frame_with_offset_np(root_pos[..., 0, :], valid_nof, block_nof).squeeze()
    # ! need to check shape of root_pos and check if offset is given correctly
    root_trf = rotmat_pos_to_trf(root_orient, root_pos)
    return root_trf

def get_aligned_target(
    inputA: np.ndarray,
    inputB: np.ndarray,
    dest_root_trf: np.ndarray,
    target_datarep: dict,
    skel: Skeleton,
):
    '''
    Given two input motion blocks, get aligned target sequence to be input to network

    Args:
        blockA: First motion block [batch, block_nof + 1, noj, 4, 4]
        blockB: Second motion block [batch, block_nof + 1, noj, 4, 4]
        next_root_trf: Next root transformation from guide, blockB will be aligned to this trf [batch, 4, 4]
        target_datarep: Data representation for target
        skel: Skeleton of character

    Returns:
        target: [nof, feat_target]
    '''
    # initialize target with first block
    inputA_feat = motion_to_feat(inputA, target_datarep, skel)
    target = inputA_feat[..., :-1, :] # make length to block_nof

    # align blockB --> next root trf squeeze 안하면 안되나?
    inputB_aligned = transform_motion_to_batch(inputB.copy(), dest_root_trf) # find closest orientation
    inputB_aligned_feat = motion_to_feat(inputB_aligned, target_datarep, skel)
    target = np.concatenate([target, inputB_aligned_feat[..., :-1, :]], axis=-2)

    return target

def pad_target_inbetween(
    target: np.ndarray,
    max_len: int,
    block_nof: int,
    valid_nof: np.ndarray
):
    # pad with zero in the middle
    batch_size = target.shape[0]
    target_padded = np.zeros([batch_size, max_len, target.shape[-1]])
    target_padded[..., :block_nof, :] = target[..., :block_nof, :]

    start_idx = np.expand_dims(valid_nof-block_nof, axis=-1)
    offset_indices = np.arange(block_nof)
    range_indices = start_idx + offset_indices
    batch_indices = np.arange(batch_size).reshape(-1, 1)
    batch_indices = np.broadcast_to(batch_indices, range_indices.shape).copy() # [batch_size, block_nof]
    target_padded[batch_indices, range_indices] = target[..., -block_nof:, :]

    return target_padded
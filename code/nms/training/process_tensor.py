import torch
import torch.nn.functional as F
from nms.motion.motion import Skeleton
from nms.rotation.rotation_tensor import aaxis_to_rotmat, rotmat_to_trf, ortho6d_to_rotmat, rotmat_to_quat, quat_closest
from nms.training.process import recon_orient_from_disp_t, recon_pos_from_disp_t

def to_trf( # to_trf using root relative joint orientation
    feat: torch.Tensor,
    feat_start_idx: dict,
    feat_end_idx: dict,
    skel: Skeleton,
    padding_mask: torch.Tensor = None,
    pad_last_frame: bool = False
):
    assert skel.parent_index_list is not None, "Parent index list should be computed in advance"
    parent_index = torch.tensor(skel.parent_index_list, dtype=torch.int32).to(feat.device) # includes root's parent too

    pad = (1, 1)
    root_init_h = feat[..., :1, feat_start_idx["root_height"]:feat_end_idx["root_height"]].unsqueeze(-2)
    root_init_pos = F.pad(root_init_h, pad)

    root_height = feat[..., feat_start_idx["root_height"]:feat_end_idx["root_height"]].unsqueeze(-2)
    root_trans_disp = feat[..., feat_start_idx["root_trans_disp"]:feat_end_idx["root_trans_disp"]].unsqueeze(-2)
    root_pos_accum = recon_pos_from_disp_t(root_trans_disp, root_init_pos)
    root_pos = torch.cat([root_pos_accum[..., :-1, :, 0:1], root_height, root_pos_accum[..., :-1, :, 2:3]], dim=-1)

    # get root trf
    root_orient = feat[..., feat_start_idx["root_orient"]:feat_end_idx["root_orient"]].unsqueeze(-2)
    root_orient = ortho6d_to_rotmat(root_orient, padding_mask)
    root_trf = rotmat_to_trf(root_orient, root_pos)

    joint_orient_root_rel = feat[..., feat_start_idx["joint_orient_root_rel"]:feat_end_idx["joint_orient_root_rel"]]
    joint_orient_root_rel = joint_orient_root_rel.reshape([*joint_orient_root_rel.shape[:-1], -1, 6])
    joint_orient_root_rel = ortho6d_to_rotmat(joint_orient_root_rel, padding_mask)
    # joint_orient_root_rel = rotmat_to_trf(joint_orient_root_rel)
    joint_orient_world = torch.matmul(root_orient, joint_orient_root_rel)
    joint_orient_world = torch.cat([root_orient, joint_orient_world], dim=-3) # combine for correct indexing

    parent_orient_world = joint_orient_world[..., parent_index, :, :]
    parent_orient_world_inv = torch.inverse(parent_orient_world)
    joint_orient_parent_rel = torch.matmul(parent_orient_world_inv, joint_orient_world) # parent-relative orientation
    joint_trf = rotmat_to_trf(joint_orient_parent_rel)

    local_trf = torch.cat([root_trf, joint_trf[..., 1:, :, :]], dim=-3) # without local translation

    if pad_last_frame:
        last_root_pos = root_pos_accum[..., -1:, :, :]

        # accum orient for later use
        root_ang_disp = feat[..., feat_start_idx["root_ang_disp"]:feat_end_idx["root_ang_disp"]].unsqueeze(-2)
        root_ang_disp = ortho6d_to_rotmat(root_ang_disp, padding_mask)
        root_orient_accum = recon_orient_from_disp_t(root_ang_disp, root_orient[..., :1, :, :, :])
        last_root_orient = root_orient_accum[..., -1:, :, :, :]

        last_root_trf = rotmat_to_trf(last_root_orient, last_root_pos)
        last_frame_trf = torch.cat([last_root_trf, local_trf[..., -1:, 1:, :, :]], dim=-3)
        local_trf = torch.cat([local_trf, last_frame_trf], dim=-4)

    return local_trf

def align(
    motion_local_trf: torch.Tensor, # should be local
    dest_root_trf: torch.Tensor
):
    '''
    Args:
        motion_local_trf: [nom, nof, noj, 4, 4]

    '''
    # transform entire motion local_trf to dest_root_trf.
    # exactly matches the first root trf
    anchor = motion_local_trf[:, :1, :1] # first root of first frame
    anchor_inv = torch.inverse(anchor)
    disp = torch.matmul(dest_root_trf, anchor_inv) # batch , nof, noj

    root = motion_local_trf[..., :1, :, :]
    aligned_root = torch.matmul(disp, root) 

    motion_local_trf_aligned = torch.cat([aligned_root, motion_local_trf[..., 1:, :, :]], dim=-3)
    return motion_local_trf_aligned

def find_closest_trf_to(
    source_trf: torch.Tensor, # [batch, 4, 4]
    target_trf: torch.Tensor,
    device: torch.device = None
):
    new_target_pos = torch.cat([target_trf[..., :1, 3], source_trf[..., 1:2, 3], target_trf[..., 2:3, 3]], dim=-1)
    quat_target = rotmat_to_quat(target_trf[..., :3, :3])
    quat_source = rotmat_to_quat(source_trf[..., :3, :3])
    axis = torch.tensor([0.0, 1.0, 0.0])
    if device is not None:
        axis = axis.to(device)

    quat_nearest, theta = quat_closest(quat_target, quat_source, axis)

    axis = axis.repeat(theta.shape[0], 1)
    theta = theta.unsqueeze(-1)
    dR_aaxis = theta * axis
    dR = aaxis_to_rotmat(dR_aaxis)
    new_target_orient = torch.matmul(dR, source_trf[..., :3, :3])
    new_target_trf = rotmat_to_trf(new_target_orient, new_target_pos)
    return new_target_trf
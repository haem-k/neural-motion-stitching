import numpy as np
from nms.motion.motion_op_numpy import transform_motion_to
from nms.rotation.rotation_numpy import rotmat_to_aaxis, aaxis_to_rotmat, rotmat_to_trf

def inertial_blending(
    motion1: np.ndarray, # [nof1, noj, 4, 4]
    motion2: np.ndarray, # [nof2, noj, 4, 4]
    num_blend: int  
):
    # get pivot indices
    nof1 = motion1.shape[0]
    nof2 = motion2.shape[0]
    assert nof2 > num_blend, "Second motion length should be longer than number of blended frames"
        
    # align second motion
    target_root_trf = motion1[-1, 0] # second motion will be connected at the end of first motion
    motion2_aligned = transform_motion_to(motion2, target_root_trf) # pivot will be first pose of second motion
    
    # root disp
    source_root_trf = motion2_aligned[0, 0]
    disp_root = target_root_trf @ np.linalg.inv(source_root_trf)
    disp_root_rot = disp_root[:3, :3]
    disp_root_pos = disp_root[:3, 3]
    
    ## joint local rot disp
    noj = motion1.shape[1]
    target_joint_rot = motion1[-1, 1:, :3, :3]
    source_joint_rot = motion2_aligned[0, 1:, :3, :3]
    source_joint_rot_inv = np.transpose(source_joint_rot, (0, 2, 1))
    disp_joint_rot = target_joint_rot @ source_joint_rot_inv
    
    
    '''
    vectorize
    '''
    weights = 1.0 - (np.arange(1, num_blend)/num_blend)
    
    # scale root disp
    disp_root_rot_aaxis = rotmat_to_aaxis(disp_root_rot)
    disp_root_rot_scaled = weights[:, None] * disp_root_rot_aaxis
    disp_root_rot_scaled = aaxis_to_rotmat(disp_root_rot_scaled)
    disp_root_pos_scaled = weights[:, None] * disp_root_pos # [9, 3]
    
    disp_root_rot_scaled_trf = rotmat_to_trf(disp_root_rot_scaled)
    
    new_poses = motion2[1:].copy()
    new_root = disp_root_rot_scaled_trf @ new_poses[:num_blend-1, 0]
    new_root_pos = disp_root_pos_scaled + new_root[:, :3, 3]
    new_root[:, :3, 3] = new_root_pos
    new_poses[:num_blend-1, 0] = new_root
    
    disp_aaxis = rotmat_to_aaxis(disp_joint_rot)
    disp_scaled = weights[:, None, None] * disp_aaxis
    disp_scaled = aaxis_to_rotmat(disp_scaled)
    new_poses[:num_blend-1, 1:, :3, :3] = disp_scaled @ new_poses[:num_blend-1, 1:, :3, :3]
    motion_blended = np.concatenate([motion1.copy(), new_poses], axis=0)
    
    return motion_blended
import numpy as np
from ..rotation.rotation_numpy import rotmat_to_quat, quat_closest, aaxis_to_rotmat, trf_to_rotmat_pos, normalize, rotmat_to_aaxis
from .motion import Pose, Skeleton

'''
Operations with only pure numpy ndarray, related to motion data
'''

def fk(motion: np.ndarray, skel: Skeleton):
    '''
    Given motion represented in local (should include rotation and translation),
    compute global trfs
    '''
    noj = motion.shape[-3]
    trfs = np.zeros_like(motion)
    for i in range(noj):
        if i == 0:
            # trfs.append(motion[..., 0, :, :])
            trfs[..., 0, :, :] = motion[..., 0, :, :]
        else:
            parent_idx = skel.get_index_joint(skel.get_joint(i).parent_joint)
            parent_trf = trfs[..., parent_idx, :, :]
            joint_trf = parent_trf @ motion[..., i, :, :]
            # trfs.append(joint_trf)
            trfs[..., i, :, :] = joint_trf
    # trfs = np.stack(trfs, axis=1)
    return trfs

def find_closest_trf(
    source_trf: np.ndarray,
    target_trf: np.ndarray,
) -> np.ndarray:
    '''
    Find new trf closest to target_trf
    by only translating source_trf on xz-plane and rotating around y-axis

    Args:
        source_trf: source transformation [4, 4]
        target_trf: target transformation [4, 4]
    '''
    # Match only x, z plane
    new_target_trf = target_trf.copy()
    new_target_trf[1, 3] = source_trf[1, 3]

    # Find nearest orientation to target_trf
    quat_target = rotmat_to_quat(target_trf[:3, :3])
    quat_pivot = rotmat_to_quat(source_trf[:3, :3])
    up_axis = np.array([0.0, 1.0, 0.0])
    _, theta = quat_closest(quat_target, quat_pivot, up_axis)

    dR_aaxis = theta * up_axis
    dR = aaxis_to_rotmat(dR_aaxis)
    new_target_trf[:3, :3] = np.matmul(dR, source_trf[:3, :3])

    return new_target_trf

def find_closest_trf_batch(
    source_trf: np.ndarray,
    target_trf: np.ndarray,
) -> np.ndarray:
    '''
    Find new trf closest to target_trf
    by only translating source_trf on xz-plane and rotating around y-axis

    Args:
        source_trf: source transformation [batch, 4, 4]
        target_trf: target transformation [batch, 4, 4]
    '''
    assert source_trf.shape == target_trf.shape, "Shape of two input should be the same"

    # Match only x, z plane
    new_target_trf = target_trf.copy()
    new_target_pos = np.concatenate([target_trf[..., :1, 3], source_trf[..., 1:2, 3], target_trf[..., 2:3, 3]], axis=-1)

    # Find nearest orientation to target_trf
    quat_target = rotmat_to_quat(target_trf[..., :3, :3])
    quat_source = rotmat_to_quat(source_trf[..., :3, :3])
    up_axis = np.array([0.0, 1.0, 0.0])
    _, theta = quat_closest(quat_target, quat_source, up_axis)

    if up_axis.ndim == theta.ndim: # theta is batched output
        up_axis = np.repeat(up_axis[None, :], theta.shape[0], axis=0)
        theta = np.expand_dims(theta, axis=-1)
    dR_aaxis = theta * up_axis
    dR = aaxis_to_rotmat(dR_aaxis)
    new_target_trf[..., :3, :3] = np.matmul(dR, source_trf[..., :3, :3])
    new_target_trf[..., :3, 3] = new_target_pos
    return new_target_trf

def move_pose_root_anchor(
    pose: np.ndarray,
    anchor_trf: np.ndarray,
    target_trf: np.ndarray,
    local: bool = True
) -> Pose:
    '''
    Represent pose w.r.t. anchor
    and match origin to target
    This is done in-place.

    Args:
        pose: Pose to be moved (noj, 4, 4)
        anchor_trf: reference frame to be represented [4, 4]
        target_trf: new origin of the pose [4, 4]
    '''
    anchor_inv = np.linalg.inv(anchor_trf)
    disp = np.matmul(target_trf, anchor_inv)

    new_root = np.matmul(disp, pose[0])
    pose[0] = new_root
    if not local:
        # TODO: update all global joint transformations
        raise NotImplementedError
    return pose

def transform_motion_to(
    motion: np.ndarray,
    target_trf: np.ndarray,
    pivot_idx: int = 0,
    closest: bool = True,
    local: bool = True
) -> np.ndarray:

    # Transform motion to target transformation
    # based on pivot pose
    # (hips at the pivot pose should be at target transformation)

    # Args:
    #     motion: Motion to be aligned (nof, noj, 4, 4)
    #     target_trf: target transformation to transform the motion
    #     pivot: pose index to be used as pivot
    #     closest: if true, only translate on the ground(xz plane) and find closeset y-axis rotation to match rotation from target_trf
    pivot_pose = motion[pivot_idx]
    pivot_root = pivot_pose[0].copy()

    if closest:
        target_trf = find_closest_trf(pivot_root, target_trf)

    for i in range(motion.shape[0]):
        motion[i] = move_pose_root_anchor(motion[i], pivot_root, target_trf, local)
    return motion

def transform_motion_to_batch(
    motion: np.ndarray,
    target_trf: np.ndarray,
    pivot_idx: int = 0,
    closest: bool = True
) -> np.ndarray:

    # Transform motion to target transformation
    # based on pivot pose
    # (hips at the pivot pose should be at target transformation)

    # Args:
    #     motion: Motion to be aligned. Should be represented in parent-relative [nom, nof, noj, 4, 4]
    #     target_trf: target transformation to transform the motion [nom, 4, 4]
    #     pivot: pose index to be used as pivot
    #     closest: if true, only translate on the ground(xz plane) and find closeset y-axis rotation to match rotation from target_trf

    pivot_pose = motion[..., pivot_idx, :, :, :]
    pivot_root = pivot_pose[..., 0, :, :].copy()

    if closest:
        target_trf = find_closest_trf_batch(pivot_root, target_trf)

    pivot_inv = np.linalg.inv(pivot_root)
    disp = target_trf @ pivot_inv
    disp = np.expand_dims(disp, axis=-3)
    disp = np.expand_dims(disp, axis=-3)

    root = motion[..., :1, :, :]
    root_transformed = disp @ root

    motion = np.concatenate([root_transformed, motion[..., 1:, :, :]], axis=-3)
    return motion

def get_root_rel_trf(motion: np.ndarray, local: bool = True, skel: Skeleton = None):
    root_trf = motion[..., :1, :, :]
    root_trf_inv = np.linalg.inv(root_trf)
    # root_trf_inv = root_trf_inv[:, np.newaxis, ...]
    world_jtrf = motion

    if local:
        assert skel is not None, "Skeleton info needed"
        world_jtrf = fk(motion, skel)
    root_rel_jtrf = root_trf_inv @ world_jtrf # ! need to check if properly broadcasted
    return root_rel_jtrf

def get_parent_rel_pos_from_skel(skel: Skeleton):
    parent_rel_pos = np.zeros((skel.noj(), 3))
    for joint in skel.joints:
        _, pos = trf_to_rotmat_pos(joint.offset_trf_from_parent)
        jidx = skel.get_index_joint(joint)
        parent_rel_pos[jidx] = pos
    return parent_rel_pos

def get_parent_rel_trf(motion: np.ndarray, local: bool = True, skel: Skeleton = None):
    if local:
        return motion

    assert skel is not None, "Skeleton info needed"
    noj = motion.shape[-3]
    parent_rel_trf = np.zeros_like(motion)
    for i in range(noj):
        if i == 0:
            parent_rel_trf[..., 0, :, :] = motion[..., 0, :, :]
        else:
            parent_idx = skel.get_index_joint(skel.get_joint(i).parent_joint)
            parent_trf = motion[..., parent_idx, :, :]
            parent_trf_inv = np.linalg.inv(parent_trf)
            parent_rel_joint_trf = parent_trf_inv @ motion[..., i, :, :]
            parent_rel_trf[..., i, :, :] = parent_rel_joint_trf
    return parent_rel_trf

def detect_contact(world_position, jnt_map, fps, vel_threshold=0.2):
    linear_vel_jpos = world_position[..., jnt_map, :]
    linear_vel = get_joint_vel_finite(linear_vel_jpos, fps)
    linear_vel_abs = np.abs(linear_vel)
    linear_vel_abs_sum = np.sum(linear_vel_abs, axis=-1)
    contacts = (linear_vel_abs_sum < vel_threshold)
    return contacts

def get_basis_from_forward_vec(forward: np.ndarray) -> np.ndarray:
    assert forward.shape[-1] == 3, 'The size of the last dimension of vector is not 3'

    temp_y = np.array([0.0, 1.0, 0.0])
    forward = normalize(forward) # z: forward vector

    x = np.cross(temp_y, forward, axis=-1)
    x = normalize(x)
    y = np.cross(forward, x)
    y = normalize(y)

    shape = list(forward.shape)
    rot_mat = np.zeros(shape[:-1] + [3, 3])
    rot_mat[..., :, 0] = x
    rot_mat[..., :, 1] = y
    rot_mat[..., :, 2] = forward
    return rot_mat
def get_joint_trans_disp(jpos: np.ndarray, pad: bool = True) -> np.ndarray:
    '''
    Args:
        jpos: [..., nof, noj, 3]

    Returns:
        disp: [..., nof-1, noj, 3]
    '''
    jpos_cur = jpos[..., :-1, :, :]
    jpos_next = jpos[..., 1:, :, :]
    disp = jpos_next - jpos_cur

    if pad:
        zero_pad = np.zeros_like(disp[..., -1, :, :])
        zero_pad = np.expand_dims(zero_pad, axis=-3)
        disp = np.concatenate([disp, zero_pad], axis=-3)
    return disp # [nom, nof, noj, 3]

def get_joint_vel_finite(jpos: np.ndarray, fps: int) -> np.ndarray:
    h = 1/fps # ! Should be fps !! not nof
    jpos_unsq_0 = np.expand_dims(jpos[..., 0, :, :], axis=-3)
    jpos_unsq_E = np.expand_dims(jpos[..., -1, :, :], axis=-3)
    padded_jpos = np.concatenate([jpos_unsq_0, jpos, jpos_unsq_E], axis=-3)

    # Central method
    prev_joint_pos = padded_jpos[..., :-2, :, :]
    next_joint_pos = padded_jpos[..., 2:, :, :]
    jvel = (next_joint_pos - prev_joint_pos) / (2 * h)

    return jvel # [nom, nof, noj, 3]


def get_joint_ang_disp(jorient: np.ndarray, pad: bool = True) -> np.ndarray:
    '''
    Args:
        jorient: [..., nof, noj, 3, 3]

    Returns:
        disp: [..., nof-1, noj, 3, 3]
    '''
    jorient_cur = jorient[..., :-1, :, :, :]
    jorient_next = jorient[..., 1:, :, :, :]
    disp = np.matmul(jorient_next, np.linalg.inv(jorient_cur))

    if pad:
        identity_pad = np.eye(3)
        identity_pad = np.broadcast_to(identity_pad, [*disp.shape[:-4], 1, disp.shape[-3], 3, 3])
        disp = np.concatenate([disp, identity_pad], axis=-4)
    return disp # [nom, nof, noj, 3]

def get_joint_angular_vel_finite(jrot: np.ndarray, fps: int) -> np.ndarray:
    '''
    jrot: [..., nof, noj, 3, 3]
    '''
    h = 1/fps
    jrot_unsq_E = np.expand_dims(jrot[..., -1, :, :, :], axis=-4) # pad last frame, will not be used later
    padded_jrot = np.concatenate([jrot, jrot_unsq_E], axis=-4) # [nof+1, noj, 3, 3]

    prev_jrot = padded_jrot[..., :-1, :, :, :]
    cur_jrot = padded_jrot[..., 1:, :, :, :]

    # delta = np.matmul(prev_jrot, np.linalg.inv(cur_jrot))
    delta = np.matmul(cur_jrot, np.linalg.inv(prev_jrot))
    delta = delta[..., :3, :3]
    jang_vel = rotmat_to_aaxis(delta)
    jang_vel = jang_vel / h

    return jang_vel

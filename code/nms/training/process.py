import numpy as np
from typing import Union
from tqdm import tqdm
import torch
import nms.utils.utils as utils
import nms.utils.bvh as bvh
import nms.motion.motion_op_numpy as phy
from nms.motion.motion import Motion, Skeleton
from nms.rotation.rotation_numpy import lerp, slerp, rotmat_pos_to_trf, quat_to_rotmat, rotmat_to_quat, ortho6d_to_rotmat, rotmat_to_ortho6d, pos_to_trf, aaxis_to_rotmat, quat_mult
from nms.motion.motion_op_numpy import find_closest_trf_batch, detect_contact, fk, get_root_rel_trf, get_parent_rel_trf, get_parent_rel_pos_from_skel
import nms.rotation.rotation_tensor as rot_t


def convert_bvh_to_npy(bvh_motion_dir, npy_save_dir, local=True, numerical=True) -> None:
    print("[Convert bvh to npy]")
    files = utils.list_files_in_dir(bvh_motion_dir, numerical)
    utils.check_create_dir(npy_save_dir)

    import_bar = tqdm(range(len(files)))
    import_bar.set_description("Importing motions")
    for i in import_bar:
        if not utils.is_bvh(files[i]):
            continue
        data_dir = f"{bvh_motion_dir}/{files[i]}"
        motion = bvh.load(data_dir)
        m = motion.to_matrix(local=local)
        np.save(f"{npy_save_dir}/{motion.get_name()}.npy", m)


def motion_to_feat(clip: Union[Motion, np.ndarray], datarep: dict, skel: Skeleton = None, local: bool = True) -> np.ndarray:
    '''
    Convert to VQVAE data

    Args:
        clip: a single segment from full motion capture [nof+1, noj, 4, 4]
              (need to provide one more frame in input clip because of velocity value )
        datarep: data representation options

    Returns:
        features: [nof, n_feat]
    '''
    character_config = datarep["character"]
    motion_config = datarep["motion"]
    total_dim, feat_start_idx, feat_end_idx, _, _ = utils.get_feature_dims(datarep)

    if isinstance(clip, Motion):
        world_position = clip.positions(local=False) # [nof, noj, 3]
        orient = clip.rotations() # [nof, noj, 3, 3] local rotation relative to parent
        root = clip.skel.root_joint
        root_idx = clip.skel.get_index_joint(root)
        root_rel_jpos = clip.get_root_rel_positions()
    elif isinstance(clip, np.ndarray):
        assert skel is not None, "Skeleton info is needed, also check 'local' parameter"
        if local:
            world_trfs = fk(clip, skel)
        else:
            world_trfs = clip
        world_position = world_trfs[..., :3, 3]
        root_rel_trf = get_root_rel_trf(clip, local=local, skel=skel)
        root_rel_jpos = root_rel_trf[..., :3, 3]
        orient = get_parent_rel_trf(clip, local=local, skel=skel)
        root_idx = 0 # assumes root index to be 0

    features = np.zeros([*clip.shape[:-3], total_dim]) # desired shape for feature: [nof, feat_dim]
    fps = datarep["fps"]

    if motion_config.get("root_pos", False):
        root_pos = world_position[..., root_idx, :] # [nof, 3]
        features[..., feat_start_idx["root_pos"]:feat_end_idx["root_pos"]] = root_pos

    if motion_config.get("root_height", False):
        root_height = world_position[..., root_idx, 1] # [nof,]
        root_height = np.expand_dims(root_height, axis=-1)
        features[..., feat_start_idx["root_height"]:feat_end_idx["root_height"]] = root_height

    if motion_config.get("root_orient", False):
        root_orient = orient[..., root_idx, :, :] # [nof, 3, 3]
        root_orient_6d = rotmat_to_ortho6d(root_orient) # [nof, 6]
        features[..., feat_start_idx["root_orient"]:feat_end_idx["root_orient"]] = root_orient_6d

    if motion_config.get("root_linear_vel", False):
        root_pos = world_position[..., root_idx, :]
        root_pos = np.expand_dims(root_pos, axis=-2)
        root_linear_vel = phy.get_joint_vel_finite(root_pos, fps) # [nof, 1, 3]
        features[..., feat_start_idx["root_linear_vel"]:feat_end_idx["root_linear_vel"]] = root_linear_vel.squeeze()

    if motion_config.get("root_angular_vel", False):
        root_orient = orient[..., root_idx, :, :]
        root_orient = np.expand_dims(root_orient, axis=-3)
        root_angular_vel = phy.get_joint_angular_vel_finite(root_orient, fps) # [nof, 1, 3]
        features[..., feat_start_idx["root_angular_vel"]:feat_end_idx["root_angular_vel"]] = root_angular_vel.squeeze()

    if motion_config.get("root_trans_disp", False):
        root_pos = world_position[..., root_idx, :]
        root_pos = np.expand_dims(root_pos, axis=-2)
        # root_linear_vel = phy.get_joint_vel_finite(root_pos, fps) # [nof, 1, 3]
        root_trans_disp = phy.get_joint_trans_disp(root_pos)
        features[..., feat_start_idx["root_trans_disp"]:feat_end_idx["root_trans_disp"]] = root_trans_disp.squeeze()

    if motion_config.get("root_ang_disp", False):
        root_orient = orient[..., root_idx, :3, :3]
        root_orient = np.expand_dims(root_orient, axis=-3)
        # root_angular_vel = phy.get_joint_angular_vel_finite(root_orient, fps) # [nof, 1, 3]
        root_ang_disp = phy.get_joint_ang_disp(root_orient)
        root_ang_disp_6d = rotmat_to_ortho6d(root_ang_disp) # [nof, 6]
        features[..., feat_start_idx["root_ang_disp"]:feat_end_idx["root_ang_disp"]] = root_ang_disp_6d.squeeze()

    if motion_config.get("joint_orient", False):
        jnt_map = utils.get_joint_map(character_config["total_joint"], motion_config["joint_orient"]["joint"])
        jnt_orient = orient[..., jnt_map, :, :] # [nof, mapped_noj, 3, 3]
        jnt_orient_6d = rotmat_to_ortho6d(jnt_orient) # [nof, mapped_noj, 6]
        # jnt_orient_6d = np.reshape(jnt_orient_6d, [jnt_orient_6d.shape[0], -1]) # [nof, mapped_noj*6]
        jnt_orient_6d = np.reshape(jnt_orient_6d, [*jnt_orient_6d.shape[:-2], -1])
        features[..., feat_start_idx["joint_orient"]:feat_end_idx["joint_orient"]] = jnt_orient_6d


    if motion_config.get("joint_orient_root_rel", False):
        jnt_map = utils.get_joint_map(character_config["total_joint"], motion_config["joint_orient_root_rel"]["joint"])
        jorient_root_rel = root_rel_trf[..., jnt_map, :3, :3]

        # jnt_orient = orient[..., jnt_map, :, :] # [nof, mapped_noj, 3, 3]
        jorient_root_rel = rotmat_to_ortho6d(jorient_root_rel) # [nof, mapped_noj, 6]
        # jnt_orient_6d = np.reshape(jnt_orient_6d, [jnt_orient_6d.shape[0], -1]) # [nof, mapped_noj*6]
        jorient_root_rel = np.reshape(jorient_root_rel, [*jorient_root_rel.shape[:-2], -1])
        features[..., feat_start_idx["joint_orient_root_rel"]:feat_end_idx["joint_orient_root_rel"]] = jorient_root_rel

    if motion_config.get("joint_angular_vel", False):
        jnt_map = utils.get_joint_map(character_config["total_joint"], motion_config["joint_angular_vel"]["joint"])
        jnt_orient = orient[..., jnt_map, :, :]
        jnt_angular_vel = phy.get_joint_angular_vel_finite(jnt_orient, fps) # [nof, mapped_noj, 3]
        jnt_angular_vel = np.reshape(jnt_angular_vel, [jnt_angular_vel.shape[0], -1]) # [nof, mapped_noj*3]
        features[..., feat_start_idx["joint_angular_vel"]:feat_end_idx["joint_angular_vel"]] = jnt_angular_vel

    if motion_config.get("joint_pos_root_rel", False):
        jnt_map = utils.get_joint_map(character_config["total_joint"], motion_config["joint_pos_root_rel"]["joint"])
        root_rel_jpos = root_rel_jpos[..., jnt_map, :] # [nof, mapped_noj, 3]
        # root_rel_jpos = np.reshape(root_rel_jpos, [root_rel_jpos.shape[0], -1]) # [nof, mapped_noj*3]
        root_rel_jpos = root_rel_jpos.reshape([*root_rel_jpos.shape[:-2], -1])
        features[..., feat_start_idx["joint_pos_root_rel"]:feat_end_idx["joint_pos_root_rel"]] = root_rel_jpos

    if motion_config.get("joint_linear_vel", False):
        jnt_map = utils.get_joint_map(character_config["total_joint"], motion_config["joint_linear_vel"]["joint"])
        linear_vel_jpos = world_position[..., jnt_map, :]
        linear_vel = phy.get_joint_vel_finite(linear_vel_jpos, fps)
        linear_vel = np.reshape(linear_vel, [linear_vel.shape[0], -1]) # [nof, mapped_noj*3]
        features[..., feat_start_idx["joint_linear_vel"]:feat_end_idx["joint_linear_vel"]] = linear_vel

    if motion_config.get("joint_trans_disp", False):
        jnt_map = utils.get_joint_map(character_config["total_joint"], motion_config["joint_trans_disp"]["joint"])
        jpos = world_position[..., jnt_map, :]
        # linear_vel = phy.get_joint_vel_finite(linear_vel_jpos, fps)
        joint_trans_disp = phy.get_joint_trans_disp(jpos)
        # joint_trans_disp = np.reshape(joint_trans_disp, [joint_trans_disp.shape[0], -1]) # [nof, mapped_noj*3]
        joint_trans_disp = joint_trans_disp.reshape([*joint_trans_disp.shape[:-2], -1])
        features[..., feat_start_idx["joint_trans_disp"]:feat_end_idx["joint_trans_disp"]] = joint_trans_disp

    if motion_config.get("foot_contact", False):
        jnt_map = utils.get_joint_map(character_config["total_joint"], motion_config["foot_contact"]["joint"])
        contacts = detect_contact(world_position, jnt_map, fps)
        contacts = np.reshape(contacts, [contacts.shape[0], -1])
        features[..., feat_start_idx["foot_contact"]:feat_end_idx["foot_contact"]] = contacts

    return features


'''
Postprocess
'''
def recon_root(features, datarep, accum_height=False, accum_orient=False, init_root_pos=None):
    motion_config = datarep["motion"]
    fps = datarep["fps"]
    _, feat_start_idx, feat_end_idx, _, _= utils.get_feature_dims(datarep)

    nom, nof, _, = features.shape
    root_data = np.eye(4)
    root_data = np.broadcast_to(root_data, [nom, nof, 1, *root_data.shape]).copy()
    if init_root_pos is not None:
        root_data[..., 0, 0, :3, 3] = init_root_pos

    # Reconstruct root
    if motion_config.get("root_pos", False):
        root_data[..., 0, :3, 3] = features[..., feat_start_idx["root_pos"]:feat_end_idx["root_pos"]]

    if motion_config.get("root_height", False):
        root_data[..., 0, 1, 3] = features[..., feat_start_idx["root_height"]:feat_end_idx["root_height"]].squeeze()

    if motion_config.get("root_height_ground_rel", False):
        root_data[..., 0, 1, 3] = features[..., feat_start_idx["root_height_ground_rel"]:feat_end_idx["root_height_ground_rel"]].squeeze()

    if motion_config.get("root_orient", False):
        root_orient_6d = features[..., feat_start_idx["root_orient"]:feat_end_idx["root_orient"]]
        try:
            root_orient = ortho6d_to_rotmat(root_orient_6d)
        except Exception as e:
            print(f"Cannot convert 6d to rotmat:{e}.\nSetting it to Identity") # ! need to see how this works
            root_orient = np.eye(3)
        root_data[..., 0, 0:3, 0:3] = root_orient

    if motion_config.get("root_linear_vel", False): # always use this value for x, z
        h = 1/fps
        root_linear_vel = features[..., feat_start_idx["root_linear_vel"]:feat_end_idx["root_linear_vel"]]
        root_linear_disp = root_linear_vel * h
        for i in range(nof-1):
            if accum_height:
                root_data[..., i+1, 0, 0:3, 3] = root_data[..., i, 0, 0:3, 3] + root_linear_disp[..., i, :]
            else:
                accum = root_data[..., i, 0, 0:3, 3] + root_linear_disp[..., i, :]
                root_data[..., i+1, 0, 0, 3] = accum[..., 0]
                root_data[..., i+1, 0, 2, 3] = accum[..., 2]

    if motion_config.get("root_angular_vel", False):
        h = 1/fps
        root_angular_vel = features[..., feat_start_idx["root_angular_vel"]:feat_end_idx["root_angular_vel"]]
        root_angular_vel = root_angular_vel * h
        delta = aaxis_to_rotmat(root_angular_vel)
        # delta = rotmat_to_trf(delta)
        if accum_orient:
            for i in range(nof-1):
                root_data[..., i+1, 0, :3, :3] = np.matmul(delta[..., i, :, :], root_data[..., i, 0, :3, :3])

    if motion_config.get("root_trans_disp", False):
        root_trans_disp = features[..., feat_start_idx["root_trans_disp"]:feat_end_idx["root_trans_disp"]]
        root_trans_disp = np.expand_dims(root_trans_disp, axis=-2)
        root_trans_disp = root_trans_disp[..., :-1, :, :] # exclude the last frame
        root_pos = recon_pos_from_disp(root_trans_disp, root_data[..., :1, :1, :3, 3])

        if accum_height:
            root_data[..., :, 0, :3, 3] = root_pos[..., :, 0, :] # pick root in joint dim
        else:
            root_data[..., :, 0, 0, 3] = root_pos[..., :, 0, 0]
            root_data[..., :, 0, 2, 3] = root_pos[..., :, 0, 2]


    if motion_config.get("root_ang_disp", False):
        root_ang_disp = features[..., feat_start_idx["root_ang_disp"]:feat_end_idx["root_ang_disp"]]
        root_ang_disp = ortho6d_to_rotmat(root_ang_disp)
        root_ang_disp = np.expand_dims(root_ang_disp, axis=-3)
        root_ang_disp = root_ang_disp[..., :-1, :, :, :] # exclude the last frame padded with identity
        root_orient = recon_orient_from_disp(root_ang_disp, root_data[..., :1, :1, :3, :3])
        if accum_orient:
            root_data[..., :, 0, :3, :3] = root_orient[..., :, 0, :, :] # pick root in joint dim

    return root_data



def recon_pos_from_disp(disp_trans: np.ndarray, pos_init: np.ndarray = None):
    '''
    Args:
        pos_init: initial position to start accumulating disp. will accumulate at the last frame [batch, nof1, noj, 3]
        disp_trans: translational displacement to be added to pos_init [batch, nof2, noj, 3]

    Returns:
        pos: resulting position [batch, nof1 + nof2, noj, 3]
    '''
    if pos_init is None:
        pos_init_E = np.zeros([*disp_trans.shape[:-3], 1, *disp_trans.shape[-2:]])
        pos_init = pos_init_E
    else:
        pos_init_E = pos_init[..., -1:, :, :]
    disp_trans = np.concatenate([pos_init_E, disp_trans], axis=-3)
    pos_accum = np.cumsum(disp_trans, axis=-3)

    pos = np.concatenate([pos_init, pos_accum[..., 1:, :, :]], axis=-3)
    return pos

def recon_pos_from_disp_t(disp_trans: torch.Tensor, pos_init: torch.Tensor = None):
    if pos_init is None:
        pos_init_E = torch.zeros([*disp_trans.shape[:-3], 1, *disp_trans.shape[-2:]])
        pos_init = pos_init_E
    else:
        pos_init_E = pos_init[..., -1:, :, :]
    disp_trans = torch.cat([pos_init_E, disp_trans], dim=-3)
    pos_accum = torch.cumsum(disp_trans, dim=-3)
    pos = torch.cat([pos_init, pos_accum[..., 1:, :, :]], dim=-3)
    return pos

def recon_orient_from_disp(disp_rot: np.ndarray, orient_init: np.ndarray = None):
    '''
    Args:
        orient_init: initial orientation to start accumulating disp. will accumulate at the last frame [batch, nof1, noj, 3, 3]
        disp_rot: rotational displacement to be added to orient_init [batch, nof2, noj, 3, 3]

    Returns:
        orient: resulting orientation [batch, nof1 + nof2, noj, 3, 3]
    '''
    if orient_init is None:
        orient_init_E = np.eye(3)
        orient_init_E = np.broadcast_to(orient_init_E, [*disp_rot.shape[:-4], 1, disp_rot.shape[-3], 3, 3])
        orient_init = orient_init_E
    else:
        orient_init_E = orient_init[..., -1:, :, :, :]

    # to quaternion
    disp_rot_quat = rotmat_to_quat(disp_rot)
    rot_accum_quat = rotmat_to_quat(orient_init_E)
    for i in range(disp_rot.shape[-4]):
        next_orient = quat_mult(disp_rot_quat[..., i:i+1, :, :], rot_accum_quat[..., i:i+1, :, :])
        rot_accum_quat = np.concatenate([rot_accum_quat, next_orient], axis=-3)

    # concat with init sequence
    rot_accum = quat_to_rotmat(rot_accum_quat)
    rot = np.concatenate([orient_init, rot_accum[..., 1:, :, :, :]], axis=-4)
    return rot


def recon_orient_from_disp_t(disp_rot: torch.Tensor, orient_init: torch.Tensor = None):
    '''
    Args:
        orient_init: initial orientation to start accumulating disp. will accumulate at the last frame [batch, nof1, noj, 3, 3]
        disp_rot: rotational displacement to be added to orient_init [batch, nof2, noj, 3, 3]

    Returns:
        orient: resulting orientation [batch, nof1 + nof2, noj, 3, 3]
    '''
    if orient_init is None:
        orient_init_E = torch.eye(3)
        orient_init_E = orient_init_E.expand([*disp_rot.shape[:-4], 1, disp_rot.shape[-3], 3, 3])
        orient_init = orient_init_E
    else:
        orient_init_E = orient_init[..., -1:, :, :, :]

    # to quaternion
    disp_rot_quat = rot_t.rotmat_to_quat(disp_rot)
    rot_accum_quat = rot_t.rotmat_to_quat(orient_init_E)
    for i in range(disp_rot.shape[-4]):
        next_orient = rot_t.quat_mult(disp_rot_quat[..., i:i+1, :, :], rot_accum_quat[..., i:i+1, :, :])
        rot_accum_quat = torch.cat([rot_accum_quat, next_orient], dim=-3)

    # concat with init sequence
    rot_accum = rot_t.quat_to_rotmat(rot_accum_quat)
    rot = torch.cat([orient_init, rot_accum[..., 1:, :, :, :]], axis=-4)
    return rot

def feat_to_trf(
    features: np.ndarray,
    datarep: dict,
    skel: Skeleton,
    local: bool = True,
    accum_height = False,
    accum_orient = False,
    init_root_pos = None
) -> np.ndarray:
    # ! 얘가 disp가 16개 있으면 17까지 계산되는데, 16으로 잘라서 나와

    character_config = datarep["character"]
    motion_config = datarep["motion"]
    total_joint = character_config["total_joint"]
    # features [nom, nof, feat_n]
    nom = features.shape[0]
    nof = features.shape[1]
    noj = len(total_joint)
    motion_data = np.eye(4)
    motion_data = np.broadcast_to(motion_data, [nom, nof, noj, *motion_data.shape]).copy()
    total_dim, feat_start_idx, feat_end_idx, _, _= utils.get_feature_dims(datarep)
    parent_index = np.array(skel.parent_index_list, dtype=np.int32) # includes root's parent too

    # Reconstruct root
    root_data = recon_root(features, datarep, accum_height, accum_orient, init_root_pos)

    # Reconstruct other joints
    if motion_config.get("joint_orient", False):
        jnt_map = utils.get_joint_map(total_joint, motion_config["joint_orient"]["joint"])
        orient_noj = len(motion_config["joint_orient"]["joint"])
        orient_6d = features[..., feat_start_idx["joint_orient"]:feat_end_idx["joint_orient"]]
        orient_6d = np.reshape(orient_6d, [nom, nof, orient_noj, -1])
        orient = ortho6d_to_rotmat(orient_6d)
        motion_data[..., jnt_map, 0:3, 0:3] = orient

    if motion_config.get("joint_orient_root_rel", False):
        joint_orient_root_rel = features[..., feat_start_idx["joint_orient_root_rel"]:feat_end_idx["joint_orient_root_rel"]]
        joint_orient_root_rel = joint_orient_root_rel.reshape([*joint_orient_root_rel.shape[:-1], -1, 6])
        joint_orient_root_rel = ortho6d_to_rotmat(joint_orient_root_rel)
        # joint_orient_root_rel = rotmat_pos_to_trf(joint_orient_root_rel)

        joint_orient_world = root_data[..., :3, :3] @ joint_orient_root_rel
        joint_orient_world = np.concatenate([root_data[..., :3, :3], joint_orient_world], axis=-3)
        parent_orient_world = joint_orient_world[..., parent_index, :, :]
        parent_orient_world_inv = np.linalg.inv(parent_orient_world)
        joint_orient_parent_rel = parent_orient_world_inv @ joint_orient_world # parent-relative orientation
        motion_data[..., 1:, :3, :3] = joint_orient_parent_rel[..., 1:, :3, :3]

    motion_data = np.concatenate([root_data, motion_data[:, :, 1:, ...]], axis=2) # Concat along joint axis
    parent_rel_pos = get_parent_rel_pos_from_skel(skel)
    motion_data[..., 1:, 0:3, 3] = parent_rel_pos[1:, :] # update only other joints
    if local:
        return motion_data
    motion_data = fk(motion_data, skel)
    return motion_data

def feat_to_pos(features, datarep, accum_height = False, accum_orient = False):
    # features [nom, nof, feat_n]
    motion_config = datarep["motion"]
    _, feat_start_idx, feat_end_idx, _, _= utils.get_feature_dims(datarep)

    # Reconstruct root
    root_data = recon_root(features, datarep, accum_height, accum_orient)

    # Reconstruct other joints' world position
    if motion_config.get("joint_pos_root_rel", False):
        nom = features.shape[0]
        nof = features.shape[1]
        noj = len(motion_config["joint_pos_root_rel"]["joint"])

        root_rel_pos = features[..., feat_start_idx["joint_pos_root_rel"]:feat_end_idx["joint_pos_root_rel"]]
        root_rel_pos = np.reshape(root_rel_pos, [nom, nof, noj, -1])
        root_rel_pos_trf = pos_to_trf(root_rel_pos)
        motion_data = root_data @ root_rel_pos_trf

    motion_data = np.concatenate([root_data, motion_data], axis=2) # Concat along joint axis
    motion_data = motion_data[..., 0:3, 3] # Get only position info
    return motion_data

def feat_to_root_pos(
    feat: np.ndarray,
    feat_start_idx: dict,
    feat_end_idx: dict
):
    root_trans_disp = feat[..., feat_start_idx["root_trans_disp"]:feat_end_idx["root_trans_disp"]]
    root_trans_disp = np.expand_dims(root_trans_disp, axis=-2)
    root_h = feat[..., feat_start_idx["root_height"]:feat_end_idx["root_height"]]
    root_h = np.expand_dims(root_h, axis=-2)
    root_h_init = root_h[..., :1, :, :]
    zero = np.zeros_like(root_h_init)
    root_pos_init = np.concatenate([zero, root_h_init, zero], axis=-1)
    root_pos_accum = recon_pos_from_disp(root_trans_disp, root_pos_init)
    root_pos = np.concatenate([root_pos_accum[..., :-1, :, 0:1], root_h, root_pos_accum[..., :-1, :, 2:3]], axis=-1)
    return root_pos

def feat_to_world_pos(
    feat: np.ndarray,
    feat_start_idx: dict,
    feat_end_idx: dict
):
    root_orient = feat[..., feat_start_idx["root_orient"]:feat_end_idx["root_orient"]]
    root_orient = ortho6d_to_rotmat(root_orient)
    root_pos = feat_to_root_pos(feat, feat_start_idx, feat_end_idx)

    # reconstruct root trf
    root_trf = rotmat_pos_to_trf(root_orient, root_pos)
    root_trf = np.expand_dims(root_trf, axis=-3)

    # get root relative joint positions
    joint_pos_root_rel = feat[..., feat_start_idx["joint_pos_root_rel"]:feat_end_idx["joint_pos_root_rel"]]
    joint_pos_root_rel = joint_pos_root_rel.reshape([*joint_pos_root_rel.shape[:-1], -1, 3])
    joint_pos_root_rel = pos_to_trf(joint_pos_root_rel)

    # reconstruct world position of joints
    world_pos_trf = root_trf @ joint_pos_root_rel
    world_pos = world_pos_trf[..., :3, 3]

    # combine with root
    world_pos = np.concatenate([root_pos, world_pos], axis=-2)
    return world_pos

def interp_guide(
    guideA: np.ndarray,
    guideB: np.ndarray,
    feat_start_idx: dict,
    feat_end_idx: dict
):
    # This function assumes batched data.
    # Will be reshaped to batch of 1 if not batched, squeezed later
    block_nof = guideA.shape[-2]
    if guideA.ndim == 2: # if not batched, expand dimension
        guideA = np.expand_dims(guideA, axis=0)
        guideB = np.expand_dims(guideB, axis=0)

    # select last and first frames
    guideA_E = guideA[..., -1, :]
    guideB_0 = guideB[..., 0, :]

    # organize feat in dict
    featsA = utils.feat_to_dict(guideA_E, feat_start_idx, feat_end_idx)
    featsB = utils.feat_to_dict(guideB_0, feat_start_idx, feat_end_idx)

    # angular features from 6D to rotmat
    featsA["root_orient"] = ortho6d_to_rotmat(featsA["root_orient"])
    featsB["root_orient"] = ortho6d_to_rotmat(featsB["root_orient"])
    featsA["root_ang_disp"] = ortho6d_to_rotmat(featsA["root_ang_disp"])
    featsB["root_ang_disp"] = ortho6d_to_rotmat(featsB["root_ang_disp"])

    # interpolation weights
    w = np.arange(1, block_nof + 1) / block_nof
    w = np.reshape(w, [1, block_nof, 1])

    # expand dimensions for interp frames
    expand_pos = lambda x: np.expand_dims(x, axis=-2)
    expand_rot = lambda x: np.expand_dims(x, axis=-3)

    # interpolate values
    root_h = lerp(expand_pos(featsA["root_height"]), expand_pos(featsB["root_height"]), w)
    root_trans_disp  = lerp(expand_pos(featsA["root_trans_disp"]), expand_pos(featsB["root_trans_disp"]), w)
    root_ang_disp_rotmat = slerp(expand_rot(featsA["root_ang_disp"]), expand_rot(featsB["root_ang_disp"]), w)
    root_ang_disp = rotmat_to_ortho6d(root_ang_disp_rotmat)
    joint_pos_root_rel = lerp(expand_pos(featsA["joint_pos_root_rel"]), expand_pos(featsB["joint_pos_root_rel"]), w)

    # reconstruct root_trf
    #   - find init root trf
    batch_size = featsA["root_orient"].shape[0]
    root_orient_init = featsA["root_orient"]
    zero = np.zeros_like(featsA["root_height"])
    root_pos_init = np.concatenate([zero, featsA["root_height"], zero], axis=-1)
    root_trf_init = rotmat_pos_to_trf(root_orient_init, root_pos_init)

    target_trf = np.expand_dims(np.eye(4), axis=0)
    target_trf = np.repeat(target_trf, batch_size, axis=0)
    root_trf_init_update = find_closest_trf_batch(root_trf_init, target_trf)
    root_trf_init_update = np.expand_dims(root_trf_init_update, axis=-3)
    if root_trf_init_update.ndim == 4: # not batched
        root_trf_init_update = np.expand_dims(root_trf_init_update, axis=0)

    #   - reconstruct root orient
    root_ang_disp_rotmat = expand_rot(root_ang_disp_rotmat) # noj dimension
    root_orient_accum = recon_orient_from_disp(root_ang_disp_rotmat, root_trf_init_update[..., :3, :3])
    root_orient = rotmat_to_ortho6d(root_orient_accum[..., :-1, :, :, :])
    root_orient = np.reshape(root_orient, [*root_orient.shape[:-2], -1])

    #   - reconstruct root pos
    root_trans_disp2 = expand_pos(root_trans_disp)
    root_pos_accum = recon_pos_from_disp(root_trans_disp2, root_trf_init_update[..., :3, 3])
    root_trf_accum = rotmat_pos_to_trf(root_orient_accum, root_pos_accum)

    # compute world joint pos with interpolated joint_pos_root_rel
    #   - process interpolated joint_pos_root_rel into joint world position
    joint_pos_root_relB = np.expand_dims(featsB["joint_pos_root_rel"], axis=-2)
    joint_pos_root_rel_combined = np.concatenate([joint_pos_root_rel, joint_pos_root_relB], axis=-2)
    joint_pos_root_rel_combined = np.reshape(joint_pos_root_rel_combined, [*joint_pos_root_rel_combined.shape[:-1], -1, 3])
    joint_pos_root_rel_trf = pos_to_trf(joint_pos_root_rel_combined)
    joint_pos_world_trf = root_trf_accum @ joint_pos_root_rel_trf
    joint_pos_world = joint_pos_world_trf[..., :3, 3]

    #   - compute joint trans disp
    jpos_world_cur = joint_pos_world[..., :-1, :, :]
    jpos_world_next = joint_pos_world[..., 1:, :, :]
    joint_trans_disp = jpos_world_next - jpos_world_cur
    joint_trans_disp = np.reshape(joint_trans_disp, [*joint_trans_disp.shape[:-2], -1])

    # concat final result
    guide_interp = np.concatenate([root_h, root_orient, root_trans_disp, root_ang_disp, joint_pos_root_rel, joint_trans_disp], axis=-1)
    return guide_interp.squeeze() # will reduce dimension if guide is not batched


'''
Indexing
'''

def get_last_valid_root_trf(
    trf: torch.Tensor,
    valid_nof: torch.Tensor,
    block_nof: int
):
    '''
    Get root trf from last valid frame in trf.

    Args:
        trf: motion trf in batches. includes motion with various lengths [batch, max_len, noj, 4, 4]
        valid_nof: number of valid frames for each sample in a batch [batch,]
        block_nof: number of frames in a motion block

    Returns:
        last_root_trf: root trf from last valid frame [batch, 4, 4]
    '''
    batch_size = trf.shape[0]
    # batch_idx = torch.arange(batch_size).unsqueeze(1).expand(batch_size)
    batch_idx = torch.arange(batch_size)
    last_frame_idx = valid_nof - block_nof - 1
    last_frame = trf[batch_idx, last_frame_idx]
    last_root_trf = last_frame[:, 0]
    return last_root_trf

def get_last_frame_with_offset(
    feat: torch.Tensor,
    valid_nof: torch.Tensor,
    offset_nof: int
):
    batch_size = feat.shape[0]
    batch_idx = torch.arange(batch_size)
    last_frame_idx = valid_nof - offset_nof
    last_frame = feat[batch_idx, last_frame_idx]
    return last_frame

def get_dest_frame(
    feat: torch.Tensor,
    valid_nof: torch.Tensor,
    block_nof: int
):
    dest_frame = get_last_frame_with_offset(feat, valid_nof, block_nof)
    return dest_frame

def get_last_frame_with_offset_np(
    feat: np.ndarray,
    valid_nof: np.ndarray,
    offset_nof: int
):
    '''
    Get last valid frame in feat(Numpy).
    Input data should be batched

    Args:
        feat: sequential feature vectors in batches. includes features with various lengths [batch, max_len, n_feat]
        valid_nof: number of valid frames for each sample in a batch [batch,]
        offset_nof: number of offset frames from the end of valid_nof [batch,]

    Returns:
        last_frame: last valid frame [batch, n_feat]
    '''
    batch_size = feat.shape[0]
    batch_idx = np.arange(batch_size)
    last_frame_idx = valid_nof - offset_nof
    last_frame = feat[batch_idx, last_frame_idx]
    return last_frame

def get_last_valid_block(
    feat: torch.Tensor,
    valid_nof: torch.Tensor,
    block_nof: int
):
    '''
    Get last motion block from batch of samples with max_len, zero padded

    Args:
        feat: motion feat in batches. includes motion with various lengths [batch, max_len, noj, 4, 4]
        valid_nof: number of valid frames for each sample in a batch [batch,]
        block_nof: number of frames in a motion block

    Returns:
        last_root_trf: root trf from last valid frame [batch, 4, 4]
    '''
    start_idx = valid_nof - block_nof
    offset = torch.arange(block_nof, device=feat.device).view(1, -1)
    gather_idx = start_idx.unsqueeze(1) + offset

    batch_size = feat.shape[0]
    batch_idx = torch.arange(batch_size, device=feat.device).unsqueeze(1).expand_as(gather_idx)

    last_block = feat[batch_idx, gather_idx]
    return last_block


def export_pos_feature_vis_np(features, datarep, export_dir, export_fname, flatten=True, accum_height = False, accum_orient = False):
    motion_config = datarep["motion"]
    assert "joint_pos_root_rel" in motion_config, f"Data representation does not include joint_pos_root_rel"
    utils.check_create_dir(export_dir)
    motion_pos = feat_to_pos(features, datarep, accum_height=accum_height, accum_orient=accum_orient)

    # export as np
    if flatten:
        total_frames_pos = np.reshape(motion_pos, [-1, *motion_pos.shape[-2:]])
    else:
        total_frames_pos = motion_pos
    np.save(f"{export_dir}/{export_fname}.npy", total_frames_pos)

def export_orient_feature_vis(features, datarep, skel_path, export_dir, export_fname, flatten=True, format="npy", accum_height = False, accum_orient = False):
    utils.check_create_dir(export_dir)
    skel = bvh.load(skel_path, load_motion=False).skel
    motion_trf = feat_to_trf(features, datarep, skel, accum_height=accum_height, accum_orient=accum_orient)

    # Flatten to one long motion
    nom, nof, noj, _, _ = motion_trf.shape
    if flatten:
        motion_trf = np.reshape(motion_trf, [nom*nof, noj, 4, 4])

    if format == "bvh":
        # Get Motion class
        motion = Motion.from_matrix(motion_trf, skel)
        bvh.save(motion, f"{export_dir}/{export_fname}.bvh")
    else:
        np.save(f"{export_dir}/{export_fname}.npy", motion_trf)

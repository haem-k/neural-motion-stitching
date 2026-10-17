'''
From fairmotion (https://github.com/facebookresearch/fairmotion)
'''
from __future__ import annotations
import numpy as np
import random
from copy import deepcopy
from ..rotation.rotation_numpy import trf_to_quat_pos, quat_normalize, quat_pos_to_trf
from typing import List

def get_index(index_dict, key):
    '''
    key is Joint / jname / jidx
    '''
    if isinstance(key, int):
        return key
    elif isinstance(key, str):
        return index_dict[key]
    else:
        return index_dict[key.name]

class Joint(object):
    '''
    Defines a joint. A hierarchy of joints form a skeleton.

    Joint object stores information about child/parent joints, base position
    transforms, and additional information in a dictionary.

    Attributes:
        name: Optional: Name of the joint. By default, we assign a randomized
            name to the joint.
        dof: Optional: Number of degrees of freedom. By default, we assume a
            fixed ball joint (dof=3). This information is stored in an
            unstructured `info` dictionary.
        parent_joint: Joint object that is parent in the skeleton.
            Set None if joint is root.
        offset_trf_from_parent: Transformation matrix indicating position and
            orientation of joint relative to parent in the character's base
            position. Defaults to identity rotation and zero position offset.
        xform_global: Transformation matrix indicating global position and
            orientation of the joint. xform_global is calculated automatically
            when parent joint is set.
        child_joints: List of child joints. Use `add_child_joints` to add joints
            to this list.
    '''
    def __init__(
        self,
        name=None,
        dof=3,
        offset_trf_from_parent=np.eye(4),
        parent_joint=None,
        length=None,
        # * direction, limits, axis: 0717 기준 안쓸 것 같아서 구현 안함
    ):
        self.name = name if name else f"joint_{random.getrandbits(32)}"
        self.child_joints = []
        self.index_child_joint = {}
        self.xform_global = np.eye(4)
        self.offset_trf_from_parent = offset_trf_from_parent
        self.parent_joint = self.set_parent_joint(parent_joint)
        self.info = {"dof": dof} # set ball joint by default

        self.length = length


    def add_child_joint(self, joint: Joint) -> None:
        assert isinstance(joint, Joint)
        assert joint.name not in self.index_child_joint.keys()
        self.index_child_joint[joint.name] = len(self.child_joints)
        self.child_joints.append(joint)
        joint.set_parent_joint(self)

    def set_parent_joint(self, joint: Joint) -> None:
        if joint is None:
            self.parent_joint = None
            return
        assert isinstance(joint, Joint)
        self.parent_joint = joint
        self.xform_global = np.dot(
            self.parent_joint.xform_global, self.offset_trf_from_parent
        )

class Skeleton(object):
    '''
    Defines a skeleton. A hierarchy of joints form a skeleton.

    Attributes:
        name: Optional: Name of the joint. By default, we assign "skeleton" as
            the name.
        v_up: Defines the up vector of the skeleton. Defaults to y-axis.
        v_face: Defines the facing direction of the skeleton in its base pose.
            Defaults to z-axis.
        v_up_env: Defines the up-vector of the environment in which motion data
            was recorded. Defaults to y-axis.

        Use `add_joint` method to add joints to the Skeleton object so that
        relevant attributes are populated appropriately.

        In the default setting, the skeleton and the environment use y-axis as
        the up-vector, and x-axis as the initial skeleton facing direction.
    '''
    def __init__(
        self,
        name="skeleton",
        v_up=np.array([0.0, 1.0, 0.0]),
        v_face=np.array([0.0, 0.0, 1.0]),
        v_up_env=np.array([0.0, 1.0, 0.0]),
    ):
        self.name = name
        self.joints = []
        self.index_joint = {}
        self.root_joint = None
        self.num_dofs = 0
        self.v_up = v_up
        self.v_face = v_face
        self.v_up_env = v_up_env

    def noj(self):
        return len(self.joints)

    def get_index_joint(self, key):
        return get_index(self.index_joint, key) # key <-- joint name / index

    def get_joint(self, key) -> Joint:
        return self.joints[self.get_index_joint(key)]

    def add_joint(self, joint, parent_joint):
        if parent_joint is None:
            assert self.noj() == 0
            self.root_joint = joint
        else:
            parent_joint = self.get_joint(parent_joint)
            parent_joint.add_child_joint(joint)
        self.index_joint[joint.name] = len(self.joints)
        self.joints.append(joint)
        self.num_dofs += joint.info["dof"]

    def get_jname_list(self) -> List[str]:
        return list(self.index_joint.keys())

    def get_parent_index_list(self) -> List[int]:
        parent_index_list = [0] # 0 for root joint
        for i in range(1, self.noj()):
            joint = self.get_joint(i)
            parent_joint = joint.parent_joint
            parent_index_list.append(self.get_index_joint(parent_joint))
        return parent_index_list

    def keep_parent_index_list(self):
        self.parent_index_list = self.get_parent_index_list()

    def __eq__(self, other: Skeleton) -> bool:
        if self.noj() != other.noj():
            return False
        if sorted(self.index_joint.keys()) != sorted(other.index_joint.keys()):
            return False
        return True



class Pose(object):
    '''
    Defines a pose. A list of poses forms a motion sequence.

    Each pose contains position and orientation of joints, in the form of
    transformation matrices, for a particular time step. The position and
    orientation information are relative to base pose. Pose object also
    stores skeleton information it is associated with, in order to access
    base pose joint data.

    Attributes:
        skel: Skeleton object of the character associated with the pose
        data: Pose data must be provided with an np.array of shape
            (num_joints, 4, 4). The order of joints must be the same as the
            order of joints in `skel.joints`.

    Use the `to_matrix()` to convert the pose object to numpy matrix form,
    and `from_matrix(data, skel)` to convert numpy matrix to pose object. This
    is useful for serializing/deserializing data for batch processing, or to
    create batched tensor data for ML model inputs.

    Use `get_transform(key, local)` to get joint transformation matrices, in
    either global or local (with respect to parent) form.
    '''

    def __init__(self, skel, data=None) -> None:
        assert isinstance(skel, Skeleton)
        if data is None:
            data = [np.eye(4) for _ in range(skel.noj())]
        assert skel.noj() == len(data)
        self.skel = deepcopy(skel)
        self.data = deepcopy(data)

    def get_transform(self, key, local):
        skel = self.skel
        if local:
            return self.data[skel.get_index_joint(key)]
        else:
            joint = skel.get_joint(key)
            trf = self.data[skel.get_index_joint(key)]

            parent_global = np.eye(4)
            parent = joint.parent_joint
            while parent is not None:
                parent_local = self.data[skel.get_index_joint(parent)]
                parent_global = np.dot(parent_local, parent_global)
                parent = parent.parent_joint

            trf = np.dot(parent_global, trf)
            return trf

    def set_transform(self, key, trf, local, do_ortho_norm=True):
        if local:
            trf1 = trf
        else: # global
            joint = self.skel.get_joint(key)
            if joint.parent_joint == None: # root
                trf1 = trf
            else: # others
                trf1 = np.eye(4)
                trf0 = self.get_transform(joint.parent_joint, local=False)
                trf1_rot = np.dot(np.linalg.inv(trf0), trf)
                trf1[:3, :3] = trf1_rot[:3, :3] # position only stored in Skeleton
        if do_ortho_norm:
            # Make sure it is orthonormal
            quat, pos = trf_to_quat_pos(trf1)
            quat = quat_normalize(quat)
            trf1 = quat_pos_to_trf(quat, pos)

        self.data[self.skel.get_index_joint(key)] = trf1

    def get_root_transform(self):
        root_idx = self.skel.get_index_joint(self.skel.root_joint)
        return self.get_transform(root_idx, local=False).copy()


    def get_root_rel_positions(self):
        root_trf_inv = np.linalg.inv(self.get_root_transform())
        root_rel_jpos = []
        for i in range(self.skel.noj()):
            world_jtrf = self.get_transform(i, local=False)
            root_rel_jtrf = root_trf_inv @ world_jtrf
            root_rel_jpos.append(root_rel_jtrf[0:3, 3])
        root_rel_jpos = np.stack(root_rel_jpos, axis=0)
        return root_rel_jpos

    def to_matrix(self, local=True):
        transforms = []
        for joint in self.skel.joints:
            transforms.append(self.get_transform(joint, local))
        return np.array(transforms)

    @classmethod
    def from_matrix(cls, data, skel, local=True):
        '''
        Expects pose data in transformation matrix format,
        with shape (noj, 4, 4)

        Positions should include data!
            - local=True: position relative to parent
            - local=False: position relative world origin
        '''
        noj, trf_r, trf_c = data.shape
        assert noj == skel.noj(), ("Data for all joints not provided")
        assert trf_r == 4 and trf_c == 4, (
            "Data not provided in 4x4 transformation matrix format"
        )
        pose = cls(skel)
        for jidx in range(len(skel.joints)):
            pose.set_transform(jidx, data[jidx], local)
        return pose

class Motion(object):
    '''
    Defines a motion sequence.

    Motion consists of the following components:
    - skeleton: Skeleton object containing information about joints and
        base pose of the subject.
    - poses: List of Pose objects. Each frame of motion is defined by a pose.
    Attributes:
        skel: Skeleton object of the character associated with the pose
        poses: List of pose objects, one for each frame of motion
        fps: Rendering frequency in Hz
        info: Free form dictionary to include more information about the motion
            sequence.

    Use the `to_matrix()` to convert the motion object to numpy matrix form
    with shape (num_frames, num_joints, 4, 4), and `from_matrix(data, skel)` to
    convert numpy matrix to pose object. This is useful for
    serializing/deserializing data for batch processing, or to create batched
    tensor data for ML model inputs.
    '''

    def __init__(self, name="motion", skel=None, fps=60) -> None:
        self.name: str = name
        self.skel: Skeleton = skel
        self.poses: List[Pose] = []
        self.fps: float = fps
        self.fps_inv: float = 1.0 / fps
        self.info: dict = {}

    def clear(self) -> None:
        self.poses = []
        self.info = {}

    def set_fps(self, fps):
        self.fps = fps
        self.fps_inv = 1.0 / fps

    def set_skeleton(self, skel: Skeleton) -> None:
        self.skel = skel
        for idx in range(len(self.poses)):
            self.poses[idx].set_skeleton(skel)

    def set_name(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name

    def get_pose_by_frame(self, frame: int) -> Pose:
        assert frame < self.nof(), f"{frame} vs. {self.nof()}"
        return self.poses[frame]

    def nof(self) -> int:
        return len(self.poses)

    def rotations(self, local: bool = True) -> np.ndarray:
        return self.to_matrix(local)[..., :3, :3]

    def positions(self, local: bool = True) -> np.ndarray:
        return self.to_matrix(local)[..., :3, 3]

    def get_root_rel_positions(self) -> np.ndarray:
        data = []
        for pose in self.poses:
            data.append(pose.get_root_rel_positions())
        return np.array(data)

    def add_one_frame(self, pose) -> None:
        if isinstance(pose, np.ndarray) or isinstance(pose, list):
            self.poses.append(Pose(self.skel, pose)) # if raw pose data is given as ndarray
        elif isinstance(pose, Pose):
            self.poses.append(pose) # if Pose class is given

    # def get_subsequence(self, start_idx: int, seq_len: int) -> Motion:

    def to_matrix(self, local: bool = True) -> np.ndarray:
        data = []
        for pose in self.poses:
            data.append(pose.to_matrix(local))
        return np.array(data)

    @classmethod
    def from_matrix(cls, data: np.ndarray, skel: Skeleton, local: bool = True, fps: float = None):
        assert data.ndim == 4, (
            "Data must be provided in transformation matrix format, with shape"
            " (seq_len, num_joints, 4, 4)"
        )
        seq_len, noj, T_0, T_1 = data.shape
        assert (
            noj == skel.noj()
        ), "Data for all joints not provided"
        assert T_0 == 4 and T_1 == 4, (
            "Data not provided in 4x4 transformation matrix format. Use "
            "fairmotion.utils.constants.eye_T() for template identity "
            "matrix"
        )

        if fps is None:
            fps = 60

        init_skel = deepcopy(skel)
        init_data = deepcopy(data)
        motion = cls(skel=init_skel, fps=fps)

        for pose_data in init_data:
            pose = Pose.from_matrix(pose_data, skel, local)
            motion.poses.append(pose)
        return motion

'''
From fairmotion (https://github.com/facebookresearch/fairmotion)
'''
import io
from typing import List
import numpy as np
from ..motion.motion import Joint, Skeleton, Motion
from ..rotation.rotation_numpy import pos_to_trf, deg_to_rad, rotmat_to_trf, angle_x_to_rotmat, angle_y_to_rotmat, angle_z_to_rotmat, trf_to_rotmat_pos, rotmat_to_euler

def load(
    file: str,
    motion: Motion = None, # ! 애가 뭐하는거지? 원래 있던 객체를 업데이트 할 수 있도록 해놨나봐
    scale: float = 0.01, # data processed in MotionBuilder
    load_skel: bool = True,
    load_motion: bool = True,
    # * load_velocity: 0717 기준 안 쓸 것 같아서 구현 안함
    v_up_skel: np.ndarray = np.array([0.0, 1.0, 0.0]),
    v_face_skel: np.ndarray = np.array([0.0, 0.0, 1.0]),
    v_up_env: np.ndarray = np.array([0.0, 1.0, 0.0]),
):
    if not motion:
        motion = Motion()
    words = None
    with open(file, "rb") as f:
        words = [word.decode() for line in f for word in line.split()]
        f.close()
    assert words is not None and len(words) > 0
    cnt = 0
    total_depth = 0
    joint_stack = [None, None]
    joint_list = []
    parent_joint_list = []

    if load_skel:
        assert motion.skel is None
        motion.skel = Skeleton(
            v_up=v_up_skel, v_face=v_face_skel, v_up_env=v_up_env
        )

    if load_skel:
        while cnt < len(words):
            joint_cur = joint_stack[-1]
            word = words[cnt].lower()
            if word == "root" or word == "joint":
                parent_joint_list.append(joint_cur)
                jname = words[cnt + 1]
                joint = Joint(name=jname)
                joint_stack.append(joint)
                joint_list.append(joint)
                cnt += 2
            elif word == "offset":
                x, y, z = (
                    float(words[cnt + 1]),
                    float(words[cnt + 2]),
                    float(words[cnt + 3]),
                )
                trf1 = pos_to_trf(scale * np.array([x, y, z]))
                joint_cur.offset_trf_from_parent = trf1
                cnt += 4
            elif word == "channels":
                ndofs = int(words[cnt + 1])
                if ndofs == 6:
                    joint_cur.info["type"] = "free"
                elif ndofs == 3:
                    joint_cur.info["type"] = "ball"
                elif ndofs == 1:
                    joint_cur.info["type"] = "revolute"
                else:
                    raise Exception("Undefined")
                joint_cur.info["dof"] = ndofs
                joint_cur.info["bvh_channels"] = []
                for i in range(ndofs):
                    joint_cur.info["bvh_channels"].append(
                        words[cnt + 2 + i].lower()
                    )
                cnt += ndofs + 2
            elif word == "end":
                joint_dummy = Joint(name="END")
                joint_stack.append(joint_dummy)
                cnt += 2
            elif word == "{":
                total_depth += 1
                cnt += 1
            elif word == "}":
                joint_stack.pop()
                total_depth -= 1
                cnt += 1
                if total_depth == 0:
                    for i in range(len(joint_list)):
                        motion.skel.add_joint(
                            joint_list[i], parent_joint_list[i],
                        )
                    break
            elif word == "hierarchy":
                cnt += 1
            else:
                raise Exception(f"Unknown Token {word} at token {cnt}")

        motion.skel.keep_parent_index_list()


    if load_motion:
        assert motion.skel is not None
        assert np.allclose(motion.skel.v_up, v_up_skel)
        assert np.allclose(motion.skel.v_face, v_face_skel)
        assert np.allclose(motion.skel.v_up_env, v_up_env)

        fname = file.split('/')[-1][:-4]
        motion.set_name(fname)

        while cnt < len(words):
            word = words[cnt].lower()
            if word == "motion":
                nof = int(words[cnt + 2])
                dt = float(words[cnt + 5])
                motion.set_fps(round(1/dt))
                cnt += 6
                t = 0.0
                range_num_dofs = range(motion.skel.num_dofs)
                positions = np.zeros(
                    (nof, motion.skel.noj(), 3, 3) # ! 이게 왜 shape이 이렇게 되지?
                )
                rotations = np.zeros(
                    (nof, motion.skel.noj(), 3) # ! 이게 왜 shape이 이렇게 되지?
                )
                trf = np.zeros(
                    (nof, motion.skel.noj(), 4, 4)
                )
                trf[...] = np.eye(4)
                position_channels = {
                    "xposition": 0,
                    "yposition": 1,
                    "zposition": 2,
                }
                rotation_channels = {
                    "xrotation": 0,
                    "yrotation": 1,
                    "zrotation": 2,
                }
                for fidx in range(nof):
                    raw_values = [
                        float(words[cnt + j]) for j in range_num_dofs
                    ]
                    cnt += motion.skel.num_dofs
                    cnt_channel = 0
                    for jidx, joint in enumerate(motion.skel.joints):
                        for channel in joint.info["bvh_channels"]:
                            value = raw_values[cnt_channel]
                            if channel in position_channels:
                                value = scale * value
                                positions[fidx][jidx][
                                    position_channels[channel]
                                ][position_channels[channel]] = value # make position diagonal matrix
                            elif channel in rotation_channels:
                                value = deg_to_rad(value)
                                rotations[fidx][jidx][
                                    rotation_channels[channel]
                                ] = value
                            else:
                                raise Exception("Unknown Channel")
                            cnt_channel += 1

                for jidx, joint in enumerate(motion.skel.joints):
                    joint_cnt_channel = 0
                    for channel in joint.info["bvh_channels"]:
                        if channel in position_channels:
                            trf[:, jidx] = trf[:, jidx] @ pos_to_trf(
                                positions[
                                    :,
                                    jidx,
                                    position_channels[channel],
                                    :,
                                ]
                            )
                        elif channel == "xrotation":
                            trf[:, jidx] = trf[:, jidx] @ rotmat_to_trf(
                                angle_x_to_rotmat(
                                    rotations[
                                        :,
                                        jidx,
                                        rotation_channels[channel]
                                    ]
                                )
                            )
                        elif channel == "yrotation":
                            trf[:, jidx] = trf[:, jidx] @ rotmat_to_trf(
                                angle_y_to_rotmat(
                                    rotations[
                                        :,
                                        jidx,
                                        rotation_channels[channel]
                                    ]
                                )
                            )
                        elif channel == "zrotation":
                            trf[:, jidx] = trf[:, jidx] @ rotmat_to_trf(
                                angle_z_to_rotmat(
                                    rotations[
                                        :,
                                        jidx,
                                        rotation_channels[channel]
                                    ]
                                )
                            )
                        joint_cnt_channel += 1

                    if joint_cnt_channel < 4: # this joint does not contain position channel
                        _, pos = trf_to_rotmat_pos(joint.offset_trf_from_parent)
                        trf[:, jidx, 0:3, 3] = pos

                for i in range(nof):
                    motion.add_one_frame(list(trf[i]))
                    t += dt

            else:
                cnt += 1

        assert motion.nof() > 0

    return motion

def _write_hierarchy(
    motion: Motion,
    file: io.TextIOWrapper,
    joint: Joint,
    scale: float = 1.0,
    rot_order: str = "XYZ",
    tab: str = ""
) -> List[str]:
    def rot_order_to_str(order):
        if order == "xyz" or order == "XYZ":
            return "Xrotation Yrotation Zrotation"
        elif order == "zyx" or order == "ZYX":
            return "Zrotation Yrotation Xrotation"
        elif order == "zxy" or order == "ZXY":
            return "Zrotation Xrotation Yrotation"
        else:
            raise NotImplementedError

    joint_order = [joint.name]
    is_root_joint = joint.parent_joint is None
    if is_root_joint:
        file.write(tab + "ROOT %s\n" % joint.name)
    else:
        file.write(tab + "JOINT %s\n" % joint.name)
    file.write(tab + "{\n")

    _, pos = trf_to_rotmat_pos(joint.offset_trf_from_parent)
    pos *= scale
    file.write(tab + "\tOFFSET %f %f %f\n" % (pos[0], pos[1], pos[2]))

    if is_root_joint:
        file.write(
            tab
            + "\tCHANNELS 6 Xposition Yposition Zposition %s\n"
            % rot_order_to_str(rot_order)
        )
    else:
        file.write(tab + "\tCHANNELS 3 %s\n" % rot_order_to_str(rot_order))
    for child_joint in joint.child_joints:
        child_joint_order = _write_hierarchy(
            motion, file, child_joint, scale, rot_order, tab + "\t"
        )
        joint_order.extend(child_joint_order)

    if len(joint.child_joints) == 0:
        file.write(tab + "\tEnd Site\n")
        file.write(tab + "\t{\n")
        file.write(tab + "\t\tOFFSET %f %f %f\n" % (0.0, 0.0, 0.0))
        file.write(tab + "\t}\n")
    file.write(tab + "}\n")
    return joint_order

def save(motion: Motion, filedir: str, scale: float = 100.0, rot_order: str = "ZXY", verbose: bool = False):
    if verbose:
        print(" >  >  Save BVH file: %s" % filedir)
    with open(filedir, "w") as f:
        '''
        Write hierarchy
        '''
        if verbose:
            print(" >  >  >  >  Write BVH hierarchy")
        f.write("HIERARCHY\n")
        joint_order = _write_hierarchy(
            motion, f, motion.skel.root_joint, scale, rot_order
        )
        '''
        Write data
        '''
        if verbose:
            print(" >  >  >  >  Write BVH data")
        t_start = 0
        dt = 1.0 / motion.fps
        nof = motion.nof()
        f.write("MOTION\n")
        f.write("Frames: %d\n" % nof)
        f.write("Frame Time: %f\n" % dt)
        t = t_start

        # rot_order_idx = rot_order_to_idx(rot_order)

        for i in range(nof):
            if verbose and i % motion.fps == 0:
                print(
                    "\r >  >  >  >  %d/%d processed (%d FPS)"
                    % (i + 1, nof, motion.fps),
                    end=" ",
                )
            pose = motion.get_pose_by_frame(i)

            for jname in joint_order:
                joint = motion.skel.get_joint(jname)
                rotmat, pos = trf_to_rotmat_pos(pose.get_transform(joint, local=True))
                pos *= scale
                euler = rotmat_to_euler(rotmat, order=rot_order, degrees=True)
                if joint == motion.skel.root_joint:
                    f.write(
                        "%f %f %f %f %f %f " % (pos[0], pos[1], pos[2], euler[0], euler[1], euler[2])
                    )
                else:
                    f.write("%f %f %f " % (euler[0], euler[1], euler[2]))
            f.write("\n")
            t += dt
            if verbose and i == nof - 1:
                print(
                    "\r >  >  >  >  %d/%d processed (%d FPS)"
                    % (i + 1, nof, motion.fps)
                )
        f.close()

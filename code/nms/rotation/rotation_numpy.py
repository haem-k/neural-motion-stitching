import numpy as np

from math import pi
from scipy.spatial.transform import Rotation


def batch_auto_reshape(x, fn, shape_in, shape_out):
    reshape = x.ndim - len(shape_in) > 1
    xx = x.reshape(-1, *shape_in) if reshape else x # flatten if needed
    y = fn(xx)
    return y.reshape(x.shape[: -len(shape_in)] + shape_out) if reshape else y

def batch_dot(x, y):
    assert x.shape[-1] == y.shape[-1]
    return (x*y).sum(-1)

def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

'''
Quaternion operations
'''
# def quat_identity():
#     return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# def quat_inv(quat):
#     inv = np.concatenate([quat[..., 0:1], -quat[..., 1:]], axis=-1)
#     return inv

def quat_mult(quat0, quat1):
    x0, y0, z0, w0 = np.split(quat0, 4, axis=-1)
    x1, y1, z1, w1 = np.split(quat1, 4, axis=-1)

    # Quaternion multiplication following [x, y, z, w] convention
    res = np.concatenate([
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    ], axis=-1)

    return res



def quat_closest(quat_target, quat, axis):
    '''
        This computes optimal-in-place orientation given a target orientation quat_target
        and a geodesic curve (quat, axis). Intuitively speaking, the optimal-in-place
        orientation is the closest orientation to quat1 when we are able to rotate quat2
        along the given axis.

        Assume quat is given in the order of xyzw (following scipy)
    '''
    # ws, vs = quat_target[..., 0], quat_target[..., 1:]
    # w0, v0 = quat[..., 0], quat[..., 1:]
    ws, vs = quat_target[..., -1], quat_target[..., :3]
    w0, v0 = quat[..., -1], quat[..., :3]

    u = normalize(axis)
    u = np.broadcast_to(u, vs.shape)

    a = ws * w0 + batch_dot(vs, v0)
    b = -ws * batch_dot(u, v0) + w0 * batch_dot(vs, u) + batch_dot(vs, np.cross(u, v0, axis=-1))
    alpha = np.arctan2(a, b)

    theta1 = -2 * alpha + pi
    theta2 = -2 * alpha - pi

    # if isinstance(theta1, np.ndarray):
    if u.ndim != theta1.ndim:
        aaxis1 = theta1[..., None] * u
        aaxis2 = theta2[..., None] * u
    else:
        aaxis1 = theta1 * u
        aaxis2 = theta2 * u

    G1 = aaxis_to_quat(aaxis1)
    G2 = aaxis_to_quat(aaxis2)

    comp_angle = batch_dot(quat_target, G1) > batch_dot(quat_target, G2)
    theta = np.where(comp_angle == True, theta1, theta2)

    q_nearest = quat_mult(G1, quat)
    return q_nearest, theta



'''
From euler to others
'''

'''
From quaternion to others
'''
def quat_to_rotmat(quat):
    rotmat = batch_auto_reshape(
        quat, lambda x: Rotation.from_quat(x).as_matrix(), (4,), (3, 3),
    )
    return rotmat

'''
From 6D representation [Zhou et al.] to others
'''
def ortho6d_to_rotmat(ortho6d):
    # Convert 6d representation to rotation matrix
    # ortho6d: [batch, nof, noj, 6]
    # rot_mat: [batch, nof, noj, 3, 3]
    x_raw = ortho6d[..., :3]
    y_raw = ortho6d[..., 3:]
    x = normalize(x_raw)
    y = normalize(y_raw)
    z_raw = np.cross(x, y, axis=-1)
    z = normalize(z_raw)
    rotmat = np.stack((x, y, z), axis=-1)
    return rotmat

'''
From angle axis to others
'''
def aaxis_to_quat(aaxis):
    quat = batch_auto_reshape(
        aaxis, lambda x: Rotation.from_rotvec(x).as_quat(), (3,), (4,),
    )
    return quat

def aaxis_to_rotmat(aaxis):
    rotmat = batch_auto_reshape(
        aaxis, lambda x: Rotation.from_rotvec(x).as_matrix(), (3,), (3, 3),
    )
    return rotmat

'''
From rotation matrix to others
'''
def rotmat_pos_to_trf(rotmat, pos):
    input_shape = rotmat.shape[:-2] if rotmat.ndim > 2 else pos.shape[:-1]
    R_flat = rotmat.reshape((-1, 3, 3))
    p_flat = pos.reshape((-1, 3))
    trf = np.zeros((int(np.prod(input_shape)), 4, 4))
    trf[...] = np.eye(4)
    trf[..., :3, :3] = R_flat
    trf[..., :3, 3] = p_flat
    return trf.reshape(list(input_shape) + [4, 4])

def rotmat_to_aaxis(rotmat):
    aaxis = batch_auto_reshape(
        rotmat, lambda x: Rotation.from_matrix(x).as_rotvec(), (3, 3), (3,),
    )
    return aaxis

def rotmat_to_ortho6d(rotmat):
    # Convert rotation matrix to 6d representation
    # rot_mat: [batch, nof, noj, 3, 3]
    # ortho6d: [batch, nof, noj, 6]
    ortho6d_x = rotmat[..., :3, 0]
    ortho6d_y = rotmat[..., :3, 1]
    ortho6d = np.concatenate((ortho6d_x, ortho6d_y), axis=-1)
    return ortho6d

def rotmat_to_quat(rotmat):
    quat = batch_auto_reshape(
        rotmat, lambda x: Rotation.from_matrix(x).as_quat(), (3, 3), (4,),
    )
    return quat

def rotmat_to_trf(rotmat):
    return rotmat_pos_to_trf(rotmat, np.array([0.0, 0.0, 0.0]))


'''
Others
'''

def pos_to_trf(pos):
    return rotmat_pos_to_trf(np.eye(3), pos)


'''
Operation
'''
def slerp(rotmat1, rotmat2, t):
    disp_mat = np.matmul(np.linalg.inv(rotmat1), rotmat2)
    scaled_disp_aaxis = t * rotmat_to_aaxis(disp_mat)
    scaled_disp_mat = aaxis_to_rotmat(scaled_disp_aaxis)
    res_mat = np.matmul(rotmat1, scaled_disp_mat)
    return res_mat

def lerp(vec1, vec2, t):
    return vec1 + (vec2 - vec1) * t

def quat_normalize(quat):
    res = quat.copy()
    # norm = np.linalg.norm(res)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise Exception("Invalid input with zero length")
    res /= norm
    return res


def rotmat_to_euler(rotmat, order="XYZ", degrees=False):
    return batch_auto_reshape(
        rotmat,
        lambda x: Rotation.from_matrix(x).as_euler(order, degrees=degrees),
        (3, 3),
        (3,),
    )

def trf_to_rotmat_pos(trf):
    rotmat = trf[..., :3, :3]
    pos = trf[..., :3, 3]
    return rotmat, pos

def trf_to_quat_pos(trf):
    rotmat, pos = trf_to_rotmat_pos(trf)
    quat = rotmat_to_quat(rotmat)
    return quat, pos

def angle_x_to_rotmat(theta):
    """
    Convert (axis) angle along x axis Ax to rotation matrix R
    """
    if isinstance(theta, np.ndarray):
        x = np.zeros((theta.shape + (3,)))
    else:
        x = np.zeros(3)
    x[..., 0] = theta
    return aaxis_to_rotmat(x)

def angle_y_to_rotmat(theta):
    """
    Convert (axis) angle along y axis Ay to rotation matrix R
    """
    if isinstance(theta, np.ndarray):
        rotmat = np.zeros(theta.shape + (3, 3))
        rotmat[...] = np.eye(3)
    else:
        rotmat = np.eye(3)
    c = np.cos(theta)
    s = np.sin(theta)
    rotmat[..., 0, 0] = c
    rotmat[..., 0, 2] = s
    rotmat[..., 2, 0] = -s
    rotmat[..., 2, 2] = c
    return rotmat

def angle_z_to_rotmat(theta):
    """
    Convert (axis) angle along z axis Az to rotation matrix R
    """
    if isinstance(theta, np.ndarray):
        rotmat = np.zeros(theta.shape + (3, 3))
        rotmat[...] = np.eye(3)
    else:
        rotmat = np.eye(3)
    c = np.cos(theta)
    s = np.sin(theta)
    rotmat[..., 0, 0] = c
    rotmat[..., 0, 1] = -s
    rotmat[..., 1, 0] = s
    rotmat[..., 1, 1] = c
    return rotmat

def quat_pos_to_trf(quat, pos):
    rotmat = quat_to_rotmat(quat)
    trf = rotmat_pos_to_trf(rotmat, pos)
    return trf

def deg_to_rad(degree):
    return degree * np.pi / 180.0

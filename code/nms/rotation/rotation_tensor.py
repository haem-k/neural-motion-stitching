import torch
import torch.nn.functional as F
from math import pi
from typing import Tuple


def batch_dot(x: torch.Tensor, y: torch.Tensor):
    assert x.shape[-1] == y.shape[-1]
    return (x*y).sum(-1)

'''
Quaternion
'''
def quat_mult(q0, q1):
    i0, j0, k0, r0 = torch.split(q0, 1, dim=-1)
    i1, j1, k1, r1 = torch.split(q1, 1, dim=-1)

    res = torch.cat([
        r0*i1 + i0*r1 + j0*k1 - k0*j1,
        r0*j1 - i0*k1 + j0*r1 + k0*i1,
        r0*k1 + i0*j1 - j0*i1 + k0*r1,
        r0*r1 - i0*i1 - j0*j1 - k0*k1
    ], dim=-1)
    return res

def quat_closest(quat1: torch.Tensor, quat2: torch.Tensor, axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        This computes optimal-in-place orientation given a target orientation quat1
        and a geodesic curve (quat2, axis). Intuitively speaking, the optimal-in-place
        orientation is the closest orientation to quat1 when we are able to rotate quat2
        along the given axis.

        Assume quat is given in the order of xyzw.
    '''
    # ws, vs = quat1[..., 0], quat1[..., 1:]
    # w0, v0 = quat2[..., 0], quat2[..., 1:]
    ws, vs = quat1[..., -1], quat1[..., :3]
    w0, v0 = quat2[..., -1], quat2[..., :3]

    u = torch.nn.functional.normalize(axis, 2.0, -1)
    u = u.expand_as(vs)

    a = ws * w0 + batch_dot(vs, v0)
    b = -ws * batch_dot(u, v0) + w0 * batch_dot(vs, u) + batch_dot(vs, torch.cross(u, v0, dim=-1))
    alpha = torch.atan2(a, b)

    theta1 = -2 * alpha + pi
    theta2 = -2 * alpha - pi

    if u.dim() != theta1.dim():
        aaxis1 = theta1[..., None] * u
        aaxis2 = theta2[..., None] * u
    else:
        aaxis1 = theta1 * u
        aaxis2 = theta2 * u

    G1 = aaxis_to_quat(aaxis1)
    G2 = aaxis_to_quat(aaxis2)

    comp_angle = batch_dot(quat1, G1) > batch_dot(quat1, G2)
    theta = torch.where(comp_angle == True, theta1, theta2)

    q_nearest = quat_mult(G1, quat2)
    return q_nearest, theta

def quat_to_rotmat(quat):
    two_s = 2.0 / torch.sum(quat * quat, dim=-1) # (...,)
    i, j, k, r = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    rotmat = torch.stack([
        1.0 - two_s * (j*j + k*k),
        two_s * (i*j - k*r),
        two_s * (i*k + j*r),
        two_s * (i*j + k*r),
        1.0 - two_s * (i*i + k*k),
        two_s * (j*k - i*r),
        two_s * (i*k - j*r),
        two_s * (j*k + i*r),
        1.0 - two_s * (i*i + j*j)
    ], dim=-1)
    return rotmat.reshape(quat.shape[:-1] + (3, 3)) # (..., 3, 3)

def rotmat_to_quat(rotmat):
    batch_dim = rotmat.shape[:-2]
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = torch.unbind(rotmat.reshape(batch_dim + (9,)), dim=-1)

    def _to_positive_sqrt(x):
        ret = torch.zeros_like(x)
        positive = x > 0
        ret[positive] = torch.sqrt(x[positive])
        return ret

    Q_square = torch.stack([
        (1.0 + R00 + R11 + R22), # 4*r*r
        (1.0 + R00 - R11 - R22), # 4*i*i
        (1.0 - R00 + R11 - R22), # 4*j*j
        (1.0 - R00 - R11 + R22), # 4*k*k
    ], dim=-1) # (..., 4)
    Q_abs = _to_positive_sqrt(Q_square) # 2*|r|, 2*|i|, 2*|j|, 2*|k|
    r, i, j, k = torch.unbind(Q_abs, dim=-1)

    Q_candidates = torch.stack([
        torch.stack([r*r, R21-R12, R02-R20, R10-R01], dim=-1),
        torch.stack([R21-R12, i*i, R01+R10, R02+R20], dim=-1),
        torch.stack([R02-R20, R01+R10, j*j, R12+R21], dim=-1),
        torch.stack([R10-R01, R02+R20, R12+R21, k*k], dim=-1),
    ], dim=-2) # (..., 4, 4)
    Q_candidates = Q_candidates / (2 * Q_abs[..., None] + 1e-8)

    Q_idx = torch.argmax(Q_square, dim=-1)
    Q = torch.gather(Q_candidates, dim=-2, index=Q_idx[..., None, None].expand(batch_dim + (1, 4))).squeeze(-2)
    Q = F.normalize(Q, dim=-1)

    quat = Q.reshape(batch_dim + (4,))
    quat = torch.cat([quat[..., 1:], quat[..., :1]], dim=-1)
    return quat


'''
6D representation [Zhou et al.]
'''
def ortho6d_to_rotmat(ortho6d: torch.Tensor, mask: torch.Tensor = None):
    """
    Converts a 6D representation to a 3×3 rotation matrix using Gram-Schmidt orthogonalization.

    Args:
        ortho6d (torch.Tensor): 6D representation with shape [batch, seq_len, num_joints, 6]
        mask (torch.Tensor, optional): Boolean tensor of shape [batch, seq_len].
            If provided, invalid frames (where mask is False) are replaced with default unit vectors
            to ensure numerical stability.

    Returns:
        rotmat: Rotation matrix of shape [batch, seq_len, num_joints, 3, 3].
    """
    x_raw = ortho6d[..., :3]
    y_raw = ortho6d[..., 3:]

    if mask is not None:
        seq_mask = (~mask).float()
        seq_mask = seq_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *x_raw.shape[-2:]) # [batch, max_len, noj, 3]

        x_i = torch.tensor([1.0, 0.0, 0.0])
        y_i = torch.tensor([0.0, 1.0, 0.0])
        x_i = x_i.expand_as(x_raw).to(seq_mask.device)
        y_i = y_i.expand_as(y_raw).to(seq_mask.device)

        x_mask = seq_mask * x_i
        y_mask = seq_mask * y_i

        x_raw = x_raw + x_mask
        y_raw = y_raw + y_mask

    # Gram Schmidt
    x = torch.nn.functional.normalize(x_raw, dim=-1)

    y_ortho = y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x
    y = torch.nn.functional.normalize(y_ortho, dim=-1, eps=1e-8)

    z = torch.linalg.cross(x, y, dim=-1)
    rotmat = torch.stack((x, y, z), dim=-1)
    return rotmat


'''
Angle axis
'''
def _split_axis_angle(aaxis):
    angle = torch.norm(aaxis, dim=-1)
    axis = aaxis / (angle[..., None] + 1e-8)
    return angle, axis

def aaxis_to_quat(aaxis):
    angle, axis = _split_axis_angle(aaxis)

    cos = torch.cos(angle / 2)[..., None]
    sin = torch.sin(angle / 2)[..., None]
    axis_sin = axis * sin

    return torch.cat([axis_sin, cos], dim=-1)

def aaxis_to_rotmat(aaxis: torch.Tensor) -> torch.Tensor:
    # split angle and axis
    angle, axis = _split_axis_angle(aaxis)
    a0, a1, a2 = axis[..., 0], axis[..., 1], axis[..., 2]
    zero       = torch.zeros_like(a0)
    batch_dims = angle.shape

    # skew symmetric matrix
    S   = torch.stack([zero, -a2, a1, a2, zero, -a0, -a1, a0, zero], dim=-1)
    S   = S.reshape(batch_dims + (3, 3))

    # rotation matrix
    I   = torch.eye(3, dtype=torch.float32, device=aaxis.device)
    I   = I.repeat(batch_dims + (1, 1))
    sin = torch.sin(angle)[..., None, None]
    cos = torch.cos(angle)[..., None, None]

    return I + S * sin + torch.matmul(S, S) * (1 - cos)



def rotmat_to_trf(rotmat: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
    # Create the last row and column
    last_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # shape: [1, 4]
    last_col = torch.tensor([[0.0], [0.0], [0.0], [1.0]])  # shape: [4, 1]

    last_row = last_row.expand(*rotmat.shape[:-2], 1, 4).clone().to(rotmat.device)
    last_col = last_col.expand(*rotmat.shape[:-2], 4, 1).clone().to(rotmat.device)
    if pos is not None:
        pos = pos.unsqueeze(-1)
        last_col = torch.cat([pos, last_col[..., 3:, :]], dim=-2)

    # Add a column of [0, 0, 0, 1] to the 3x3 rotation matrix
    upper_3x4 = torch.cat((rotmat, last_col[..., :3, :]), dim=-1)

    # Concatenate the last row to form the final 4x4 homogeneous transformation matrix
    trf = torch.cat((upper_3x4, last_row), dim=-2)
    return trf

def pos_to_trf(pos: torch.Tensor) -> torch.Tensor:
    rotmat = torch.eye(3, device=pos.device).expand(*pos.shape[:-1], 3, 3).clone()
    return rotmat_to_trf(rotmat, pos)

# src/ensam3d_inference/core/pose_estimation/heads/mhr_head/utils.py
import torch
import torch.nn.functional as F
from .types import (
    BodyContTensor,
    BodyModelParamsTensor,
    EulerAnglesTensor,
    HandContTensor,
    HandModelParamsTensor,
    Rot6DTensor,
    RotMatTensor,
    Rot6DGeneric,
    HandParamMask
)

# ---------------------------------------------------------------------------
# Body pose index constants
# ---------------------------------------------------------------------------

_BODY_3DOF_IDXS = torch.LongTensor([
    (0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20),
    (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39),
    (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79),
    (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106),
    (114, 98, 109), (115, 99, 103), (130, 131, 132),
])
_BODY_1DOF_IDXS = torch.LongTensor([
    1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51,
    52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83,
    84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117,
    118, 119, 120, 121, 122, 123,
])
_BODY_TRANS_IDXS = torch.LongTensor([124, 125, 126, 127, 128, 129])

_MHR_PARAM_HAND_IDXS = [
    62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
    113, 114, 115,
]

_HAND_DOFS = torch.tensor([3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1])


def make_hand_param_mask() -> HandParamMask:
    """
    Create a boolean mask identifying hand joint indices in the body pose parameter vector.

    Intended to be called once during MHRHead initialisation and registered
    as a buffer via register_buffer so the mask moves to the correct device
    automatically with the model.

    Returns
    -------
    HandParamMask
        Boolean mask, shape (133,); True at the 54 indices corresponding to
        hand joints that are zeroed out before passing body pose to the MHR model.
    """
    mask = torch.zeros(133).bool()
    mask[_MHR_PARAM_HAND_IDXS] = True
    return mask


# ---------------------------------------------------------------------------
# Rotation conversion utilities
# ---------------------------------------------------------------------------

def rot6d_to_rotmat(x: Rot6DTensor) -> RotMatTensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrices.

    Parameters
    ----------
    x : Rot6DTensor
        6D rotation vectors, shape (B, 6), following Zhou et al. (CVPR 2019).

    Returns
    -------
    RotMatTensor
        Rotation matrices, shape (B, 3, 3).
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(
        a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1, dim=-1
    )
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _batch_xyz_from_6d(poses: Rot6DGeneric) -> EulerAnglesTensor:
    """
    Convert 6D rotation representation to XYZ Euler angles.

    Parameters
    ----------
    poses : Rot6DGeneric
        First two columns of a rotation matrix concatenated, shape (..., 6).

    Returns
    -------
    EulerAnglesTensor
        XYZ Euler angles in radians, shape (..., 3).
    """
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    z = torch.linalg.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.linalg.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1)

    sy = torch.sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2)
    singular = (sy < 1e-6).float()

    ex = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    ey = torch.atan2(-matrix[..., 2, 0], sy)
    ez = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])

    ex_s = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    ey_s = torch.atan2(-matrix[..., 2, 0], sy)
    ez_s = torch.zeros_like(matrix[..., 1, 0])

    out = torch.zeros_like(matrix[..., 0])
    out[..., 0] = ex * (1 - singular) + ex_s * singular
    out[..., 1] = ey * (1 - singular) + ey_s * singular
    out[..., 2] = ez * (1 - singular) + ez_s * singular
    return out


# ---------------------------------------------------------------------------
# Pose parameter conversion utilities
# ---------------------------------------------------------------------------

def compact_cont_to_model_params_hand(hand_cont: HandContTensor) -> HandModelParamsTensor:
    """
    Convert continuous hand pose representation to hand model parameters.

    Parameters
    ----------
    hand_cont : HandContTensor
        Continuous hand pose encoding, shape (B, 54).

    Returns
    -------
    HandModelParamsTensor
        Hand joint parameters in Euler angle space, shape (B, 27).
    """
    mask_cont_3dof = torch.cat(
        [torch.ones(2 * k).bool() * (k in [3]) for k in _HAND_DOFS]
    )
    mask_cont_1dof = torch.cat(
        [torch.ones(2 * k).bool() * (k in [1, 2]) for k in _HAND_DOFS]
    )
    mask_params_3dof = torch.cat(
        [torch.ones(k).bool() * (k in [3]) for k in _HAND_DOFS]
    )
    mask_params_1dof = torch.cat(
        [torch.ones(k).bool() * (k in [1, 2]) for k in _HAND_DOFS]
    )

    cont_3dof = hand_cont[..., mask_cont_3dof].unflatten(-1, (-1, 6))
    params_3dof = _batch_xyz_from_6d(cont_3dof).flatten(-2, -1)

    cont_1dof = hand_cont[..., mask_cont_1dof].unflatten(-1, (-1, 2))
    params_1dof = torch.atan2(cont_1dof[..., -2], cont_1dof[..., -1])

    hand_model_params = torch.zeros(*hand_cont.shape[:-1], 27).to(hand_cont)
    hand_model_params[..., mask_params_3dof] = params_3dof
    hand_model_params[..., mask_params_1dof] = params_1dof
    return hand_model_params


def compact_cont_to_model_params_body(body_pose_cont: BodyContTensor) -> BodyModelParamsTensor:
    """
    Convert continuous body pose representation to body model parameters.

    Parameters
    ----------
    body_pose_cont : BodyContTensor
        Continuous body pose encoding, shape (B, 260).

    Returns
    -------
    BodyModelParamsTensor
        Body pose parameters in Euler angle and translation space, shape (B, 133).
    """
    n_3dof = len(_BODY_3DOF_IDXS) * 3
    n_1dof = len(_BODY_1DOF_IDXS)

    cont_3dof = body_pose_cont[..., : 2 * n_3dof].unflatten(-1, (-1, 6))
    cont_1dof = body_pose_cont[
        ..., 2 * n_3dof : 2 * n_3dof + 2 * n_1dof
    ].unflatten(-1, (-1, 2))
    cont_trans = body_pose_cont[..., 2 * n_3dof + 2 * n_1dof :]

    params_3dof = _batch_xyz_from_6d(cont_3dof).flatten(-2, -1)
    params_1dof = torch.atan2(cont_1dof[..., -2], cont_1dof[..., -1])

    body_pose_params = torch.zeros(*body_pose_cont.shape[:-1], 133).to(body_pose_cont)
    body_pose_params[..., _BODY_3DOF_IDXS.flatten()] = params_3dof
    body_pose_params[..., _BODY_1DOF_IDXS] = params_1dof
    body_pose_params[..., _BODY_TRANS_IDXS] = cont_trans
    return body_pose_params
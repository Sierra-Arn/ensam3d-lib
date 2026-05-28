# Copyright (c) 2026 Ilya Snegov (aka Sierra Arn)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/ensam3d_inference/core/pose_estimation/heads/mhr_head/types.py
from typing import NamedTuple
import torch
from jaxtyping import Float, Bool


HandParamMask = Bool[torch.Tensor, "133"]
"""
Boolean mask for hand joint indices in the body pose parameter vector represented as a PyTorch
tensor with shape (133,) and boolean dtype, where True marks the 54 indices corresponding to
hand joints that are zeroed out in the body pose before passing to the MHR model.
"""


Rot6DTensor = Float[torch.Tensor, "B 6"]
"""
6D rotation representation represented as a PyTorch tensor with shape (B, 6) and floating-point
dtype (matching the model output), where B is the number of frames in the batch and 6 is the
continuous rotation representation from Zhou et al. (CVPR 2019), consisting of the first two
columns of a 3x3 rotation matrix concatenated into a single vector.
"""


RotMatTensor = Float[torch.Tensor, "B 3 3"]
"""
Rotation matrix represented as a PyTorch tensor with shape (B, 3, 3) and floating-point dtype
(matching the model output), where B is the number of frames in the batch and (3, 3) is a
rotation matrix in SO(3).
"""


Rot6DGeneric = Float[torch.Tensor, "... 6"]
"""
6D rotation representation for arbitrary batch dimensions represented as a PyTorch tensor
with shape (..., 6) and floating-point dtype, where the leading dimensions are arbitrary
(batch, joint count, or both) and 6 is the continuous rotation representation from Zhou
et al. (CVPR 2019) consisting of the first two columns of a 3x3 rotation matrix concatenated;
used as an intermediate representation during conversion between Euler angles and rotation
matrices for both body and hand joints.
"""


EulerAnglesTensor = Float[torch.Tensor, "B 3"]
"""
XYZ Euler angles represented as a PyTorch tensor with shape (B, 3) and floating-point dtype
(matching the model output), where B is the number of frames in the batch and 3 corresponds
to (rx, ry, rz) rotation angles in radians around the X, Y, and Z axes respectively.
"""


HandContTensor = Float[torch.Tensor, "B 54"]
"""
Continuous hand pose representation represented as a PyTorch tensor with shape (B, 54) and
floating-point dtype (matching the model output), where B is the number of frames in the batch
and 54 is the continuous encoding of 27 hand joint parameters mixing 6D representations for
3-DoF joints and sin-cos pairs for 1-DoF and 2-DoF joints.
"""


HandModelParamsTensor = Float[torch.Tensor, "B 27"]
"""
Hand model parameters represented as a PyTorch tensor with shape (B, 27) and floating-point
dtype (matching the model output), where B is the number of frames in the batch and 27 is the
number of hand joint parameters in Euler angle space ordered by joint then axis.
"""


BodyContTensor = Float[torch.Tensor, "B 260"]
"""
Continuous body pose representation represented as a PyTorch tensor with shape (B, 260) and
floating-point dtype (matching the model output), where B is the number of frames in the batch
and 260 is the continuous encoding of body joint parameters mixing 6D representations for 3-DoF
joints and sin-cos pairs for 1-DoF joints plus translation components.
"""


BodyModelParamsTensor = Float[torch.Tensor, "B 133"]
"""
Body model parameters represented as a PyTorch tensor with shape (B, 133) and floating-point
dtype (matching the model output), where B is the number of frames in the batch and 133 is the
number of body pose parameters in Euler angle and translation space after conversion from the
continuous representation.
"""


Keypoints3DTensor = Float[torch.Tensor, "B 70 3"]
"""
3D keypoints in camera space represented as a PyTorch tensor with shape (B, 70, 3) and
floating-point dtype (matching the model output), where B is the number of frames in the
batch, 70 is the fixed number of body keypoints after slicing from the full 308 Sapiens
keypoints, and 3 corresponds to (x, y, z) coordinates in camera space with y and z axes
flipped to match the target coordinate system.
"""


PoseRawTensor = Float[torch.Tensor, "B 266"]
"""
Concatenated continuous pose representation represented as a PyTorch tensor with shape (B, 266)
and floating-point dtype (matching the model output), where B is the number of frames in the
batch and 266 is the concatenation of the 6D global rotation (6) and the continuous body pose
vector (260) before conversion to Euler angles or model parameters.
"""


GlobalRotTensor = Float[torch.Tensor, "B 3"]
"""
Global orientation as ZYX Euler angles represented as a PyTorch tensor with shape (B, 3) and
floating-point dtype (matching the model output), where B is the number of frames in the batch
and 3 corresponds to (Z, Y, X) Euler angles in radians produced by roma.rotmat_to_euler.
"""


BodyPoseTensor = Float[torch.Tensor, "B 133"]
"""
MHR body pose parameters represented as a PyTorch tensor with shape (B, 133) and floating-point
dtype (matching the model output), where B is the number of frames in the batch and 133 is the
fixed number of MHR body pose parameters in Euler angle and translation space after conversion
from the continuous representation, with hand joint indices and jaw zeroed out.
"""


ShapeTensor = Float[torch.Tensor, "B 45"]
"""
Shape PCA coefficients represented as a PyTorch tensor with shape (B, 45) and floating-point
dtype (matching the model output), where B is the number of frames in the batch and 45 is the
number of shape principal components used by the MHR body model.
"""


ScaleTensor = Float[torch.Tensor, "B 28"]
"""
Scale PCA coefficients represented as a PyTorch tensor with shape (B, 28) and floating-point
dtype (matching the model output), where B is the number of frames in the batch and 28 is the
number of scale principal components used by the MHR body model.
"""


HandTensor = Float[torch.Tensor, "B 108"]
"""
Hand PCA coefficients represented as a PyTorch tensor with shape (B, 108) and floating-point
dtype (matching the model output), where B is the number of frames in the batch and 108 is the
concatenation of left (54) and right (54) hand pose PCA coefficients in the continuous
representation before conversion to joint angles.
"""


VerticesTensor = Float[torch.Tensor, "B V 3"]
"""
Skinned mesh vertices in camera space represented as a PyTorch tensor with shape (B, V, 3) and
floating-point dtype (matching the model output), where B is the number of frames in the batch,
V is the number of vertices in the MHR body mesh (18439), and 3 corresponds to (x, y, z)
coordinates in metres with y and z axes flipped to match the target coordinate system.
"""


class MHRHeadOutput(NamedTuple):
    """
    Output of MHRHead.forward.

    Attributes
    ----------
    pred_keypoints_3d : Keypoints3DTensor
        3D keypoints in camera space, shape (B, 70, 3); always populated.
    pred_pose_raw : PoseRawTensor or None
        Concatenated 6D global rotation and continuous body pose, shape (B, 266).
    global_rot : GlobalRotTensor or None
        Global orientation as ZYX Euler angles, shape (B, 3).
    body_pose : BodyPoseTensor or None
        MHR body pose parameters before hand injection, shape (B, 133).
    shape : ShapeTensor or None
        Shape PCA coefficients, shape (B, 45).
    scale : ScaleTensor or None
        Scale PCA coefficients, shape (B, 28).
    hand : HandTensor or None
        Hand PCA coefficients for left and right hands, shape (B, 108).
    pred_vertices : VerticesTensor or None
        Mesh vertices in camera space, shape (B, V, 3).
    """

    pred_keypoints_3d: Keypoints3DTensor
    pred_pose_raw: PoseRawTensor | None
    global_rot: GlobalRotTensor | None
    body_pose: BodyPoseTensor | None
    shape: ShapeTensor | None
    scale: ScaleTensor | None
    hand: HandTensor | None
    pred_vertices: VerticesTensor | None


PoseParamsTensor = Float[torch.Tensor, "B P"]
"""
Raw predicted pose parameter vector represented as a PyTorch tensor with shape (B, P) and
floating-point dtype (matching the model output), where B is the number of frames in the batch
and P is the total number of pose parameters predicted by the MLP projection (npose), combining
global rotation, continuous body pose, shape, scale, hand, and face components in a single
concatenated vector.
"""


SkinnedVertsTensor = Float[torch.Tensor, "B V 3"]
"""
Skinned mesh vertices in camera space represented as a PyTorch tensor with shape (B, V, 3) and
floating-point dtype (matching the model output), where B is the number of frames in the batch,
V is the number of vertices in the MHR body mesh (18439), and 3 corresponds to (x, y, z)
coordinates in metres after dividing the raw TorchScript output by 100.
"""
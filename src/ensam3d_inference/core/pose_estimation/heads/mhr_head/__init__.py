# src/ensam3d_inference/core/pose_estimation/heads/mhr_head/__init__.py
import roma
import torch
import torch.nn as nn
from ..layers import MLP
from ..types import PoseToken
from .types import (
    BodyContTensor,
    BodyModelParamsTensor,
    HandContTensor,
    Keypoints3DTensor,
    MHRHeadOutput,
    PoseParamsTensor,
    SkinnedVertsTensor,
)
from .utils import (
    compact_cont_to_model_params_body,
    compact_cont_to_model_params_hand,
    make_hand_param_mask,
    rot6d_to_rotmat,
)
from .....shared import DeviceType, config


class MHRHead(nn.Module):
    """
    Mesh and Human Reconstruction head predicting body pose, shape, scale,
    and hand parameters from a decoder pose token.

    Projects the pose token to a parameter vector via an MLP, converts
    continuous representations to model parameters, and runs the TorchScript
    MHR body model to produce vertices, keypoints, and joint coordinates.

    Parameters
    ----------
    mhr_model_path : str
        Filesystem path to the TorchScript MHR model file.
    device : DeviceType, optional
        Device used to load the TorchScript MHR model. Default is DeviceType.CUDA.

    Attributes
    ----------
    proj : MLP
        MLP projecting the pose token to the full pose parameter vector.
    scale_mean : torch.Tensor
        Mean scale vector, shape (68,); registered as a buffer and loaded
        from the checkpoint.
    scale_comps : torch.Tensor
        Scale PCA components, shape (28, 68); registered as a buffer and
        loaded from the checkpoint.
    faces : torch.Tensor
        Mesh face indices, shape (36874, 3); registered as a buffer and
        loaded from the checkpoint.
    hand_pose_mean : torch.Tensor
        Mean hand pose vector, shape (54,); registered as a buffer and
        loaded from the checkpoint.
    hand_pose_comps : torch.Tensor
        Hand pose PCA components, shape (54, 54); registered as a buffer
        and loaded from the checkpoint.
    hand_joint_idxs_left : torch.Tensor
        Left hand joint indices into the full pose vector, shape (27,);
        registered as a buffer and loaded from the checkpoint.
    hand_joint_idxs_right : torch.Tensor
        Right hand joint indices into the full pose vector, shape (27,);
        registered as a buffer and loaded from the checkpoint.
    keypoint_mapping : torch.Tensor
        Linear mapping from mesh vertices and joints to 308 keypoints,
        shape (308, 18566); registered as a buffer and loaded from the
        checkpoint.
    mhr_param_hand_mask : torch.Tensor
        Boolean mask for hand joint indices in the body pose vector,
        shape (133,); registered as a buffer for automatic device transfer.
    mhr : torch.jit.ScriptModule
        Frozen TorchScript MHR body model.
    device : DeviceType
        Device used for inference.
    """

    def __init__(
        self,
        mhr_model_path: str,
        device: DeviceType = DeviceType.CUDA,
    ) -> None:
        super().__init__()
        self.device = device
        self.proj = MLP(
            in_dims=config.decoder_dim,
            hidden_dims=config.decoder_dim,
            out_dims=config.npose,
            num_layers=2,
        )
        torch.nn.init.zeros_(self.proj.layers[-2].bias)

        self.register_buffer("scale_mean", torch.zeros(68))
        self.register_buffer("scale_comps", torch.zeros(28, 68))
        self.register_buffer("faces", torch.zeros(36874, 3).long())
        self.register_buffer("hand_pose_mean", torch.zeros(54))
        self.register_buffer("hand_pose_comps", torch.eye(54))
        self.register_buffer("hand_joint_idxs_left", torch.zeros(27).long())
        self.register_buffer("hand_joint_idxs_right", torch.zeros(27).long())
        self.register_buffer("keypoint_mapping", torch.zeros(308, 18439 + 127))
        self.register_buffer("mhr_param_hand_mask", make_hand_param_mask())

        self.mhr = torch.jit.load(mhr_model_path, map_location=self.device)
        for param in self.mhr.parameters():
            param.requires_grad = False

    def _replace_hands_in_pose(
        self,
        full_pose_params: BodyModelParamsTensor,
        hand_pose_params: HandContTensor,
    ) -> BodyModelParamsTensor:
        """
        Inject hand pose parameters into the full body pose vector.

        Parameters
        ----------
        full_pose_params : BodyModelParamsTensor
            Full body pose vector, shape (B, 136).
        hand_pose_params : HandContTensor
            Continuous hand pose encoding for both hands, shape (B, 108).

        Returns
        -------
        BodyModelParamsTensor
            Updated full body pose vector with hand joints replaced, shape (B, 136).
        """
        left_hand_params, right_hand_params = torch.split(
            hand_pose_params,
            [config.mhr_hand_comps, config.mhr_hand_comps],
            dim=1,
        )
        left_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean.float()
            + torch.einsum("da,ab->db", left_hand_params.float(), self.hand_pose_comps.float())
        )
        right_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean.float()
            + torch.einsum("da,ab->db", right_hand_params.float(), self.hand_pose_comps.float())
        )
        full_pose_params[:, self.hand_joint_idxs_left] = left_model_params
        full_pose_params[:, self.hand_joint_idxs_right] = right_model_params
        return full_pose_params

    def forward(
        self,
        x: PoseToken,
        init_estimate: PoseParamsTensor,
        full_output: bool = True,
    ) -> MHRHeadOutput:
        """
        Predict body mesh and keypoints from a decoder pose token.

        Parameters
        ----------
        x : PoseToken
            Pose token from the decoder, shape (B, C).
        init_estimate : PoseParamsTensor
            Additive residual initialisation for the pose parameters,
            shape (B, npose).
        full_output : bool, optional
            If True, computes and returns all pose, shape, and mesh fields.
            If False, only pred_keypoints_3d is populated and intermediate
            computations irrelevant to it are skipped. Default is True.

        Returns
        -------
        MHRHeadOutput
            pred_keypoints_3d always populated; all other fields populated
            only when full_output is True.
        """
        batch_size = x.shape[0]
        pred: PoseParamsTensor = self.proj(x) + init_estimate

        count = 6
        global_rot_6d = pred[:, :count]
        global_rot_euler = roma.rotmat_to_euler("ZYX", rot6d_to_rotmat(global_rot_6d.float()))
        global_trans = torch.zeros_like(global_rot_euler)

        pred_pose_cont: BodyContTensor = pred[:, count : count + config.mhr_body_cont_dim]
        count += config.mhr_body_cont_dim
        pred_pose_euler: BodyModelParamsTensor = compact_cont_to_model_params_body(
            pred_pose_cont.float()
        )
        pred_pose_euler[:, self.mhr_param_hand_mask] = 0
        pred_pose_euler[:, -3:] = 0

        pred_shape = pred[:, count : count + config.mhr_shape_comps]
        count += config.mhr_shape_comps
        pred_scale = pred[:, count : count + config.mhr_scale_comps]
        count += config.mhr_scale_comps
        pred_hand = pred[:, count : count + config.mhr_hand_comps * 2]
        pred_face = torch.zeros(batch_size, config.mhr_face_comps, device=x.device, dtype=x.dtype)

        scales = (
            self.scale_mean.float()[None, :]
            + pred_scale.float() @ self.scale_comps.float()
        )
        full_pose_params = torch.cat(
            [global_trans * 10, global_rot_euler.float(), pred_pose_euler.float()[..., :130]],
            dim=1,
        )
        full_pose_params = self._replace_hands_in_pose(full_pose_params, pred_hand)

        # The MHR TorchScript solver internally relies on sparse CUDA matmul/addmm
        # kernels that have no bfloat16 implementations, and its kinematic/PCA
        # vertex reconstruction requires FP32 mantissa precision for geometric
        # stability. Force strict FP32 execution for this block.
        with torch.autocast(device_type=self.device, enabled=False):
            curr_skinned_verts, curr_skel_state = self.mhr(
                pred_shape.float(),
                torch.cat([full_pose_params, scales], dim=1),
                pred_face.float(),
            )
        
        curr_joint_coords, _, _ = torch.split(curr_skel_state, [3, 4, 1], dim=2)
        curr_skinned_verts: SkinnedVertsTensor = curr_skinned_verts / 100
        curr_joint_coords = curr_joint_coords / 100

        model_vert_joints = torch.cat([curr_skinned_verts, curr_joint_coords], dim=1)
        j3d: Keypoints3DTensor = (
            (
                self.keypoint_mapping.float()
                @ model_vert_joints.float().permute(1, 0, 2).flatten(1, 2)
            )
            .reshape(-1, model_vert_joints.shape[0], 3)
            .permute(1, 0, 2)
        )[:, :70]
        j3d[..., [1, 2]] *= -1

        if not full_output:
            return MHRHeadOutput(
                pred_keypoints_3d=j3d.reshape(batch_size, -1, 3),
                pred_pose_raw=None,
                global_rot=None,
                body_pose=None,
                shape=None,
                scale=None,
                hand=None,
                pred_vertices=None,
            )

        curr_skinned_verts[..., [1, 2]] *= -1

        return MHRHeadOutput(
            pred_keypoints_3d=j3d.reshape(batch_size, -1, 3),
            pred_pose_raw=torch.cat([global_rot_6d, pred_pose_cont], dim=1),
            global_rot=global_rot_euler,
            body_pose=pred_pose_euler,
            shape=pred_shape,
            scale=pred_scale,
            hand=pred_hand,
            pred_vertices=curr_skinned_verts.reshape(batch_size, -1, 3),
        )
# src/ensam3d_inference/core/pose_estimation/__init__.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .types import (
    CroppedSizeTensor,
    AffineTransTensor,
    IteratedPoseOutput,
    PoseEstimatorInput,
    PoseEstimatorOutput,
)
from .decoder import PromptableDecoder
from .heads import MHRHead, PerspectiveHead
from .heads.layers import MLP
from .heads.mhr_head.types import Keypoints3DTensor
from .heads.perspective_head.types import PerspectiveHeadOutput
from .utils import full_to_crop, get_decoder_condition
from .utils.position_encoding import PositionEmbeddingRandom
from .utils.projection import perspective_projection
from ..feature_extraction.backbone.types import BackboneOutput, TokenSequence
from ...shared import DeviceType, config


class PoseEstimator(nn.Module):
    """
    Iterative pose estimator combining a transformer decoder with MHR and
    camera heads.

    Takes ray-conditioned image embeddings from FeatureExtractor and produces
    full-body mesh, keypoints, and camera parameters via a six-layer decoder
    with iterative keypoint token refinement.

    Parameters
    ----------
    mhr_model_path : str
        Filesystem path to the TorchScript MHR model file.
    device : DeviceType, optional
        Device used to load the TorchScript MHR model. Default is DeviceType.CUDA.

    Attributes
    ----------
    mhr_head : MHRHead
        Mesh regression head predicting body pose, shape, and vertices.
    camera_head : PerspectiveHead
        Weak-perspective camera head predicting (s, tx, ty).
    decoder : PromptableDecoder
        Cross-attention transformer decoder stack.
    image_pe : PositionEmbeddingRandom
        Non-learnable positional encoding for the image context sequence.
    init_pose : nn.Embedding
        Learnable initial pose token, shape (1, npose).
    init_camera : nn.Embedding
        Learnable initial camera token, shape (1, NCAM).
    init_to_token : nn.Linear
        Projects (COND_DIM + npose + NCAM) to DEC_DIM for the first pass.
    prev_to_token : nn.Linear
        Projects (npose + NCAM) to DEC_DIM for iterative correction.
    keypoint_embedding : nn.Embedding
        Learnable 2D keypoint type embeddings, shape (N_KEYPOINTS, DEC_DIM).
    keypoint_posemb_linear : MLP
        Projects 2D keypoint coordinates to DEC_DIM.
    keypoint_feat_linear : nn.Linear
        Projects image features sampled at keypoint locations to DEC_DIM.
    keypoint3d_embedding : nn.Embedding
        Learnable 3D keypoint type embeddings, shape (N_KEYPOINTS, DEC_DIM).
    keypoint3d_posemb_linear : MLP
        Projects pelvis-centred 3D keypoint coordinates to DEC_DIM.
    """

    def __init__(
        self,
        mhr_model_path: str,
        device: DeviceType = DeviceType.CUDA,
    ) -> None:
        super().__init__()

        self.mhr_head = MHRHead(mhr_model_path=mhr_model_path, device=device)
        self.camera_head = PerspectiveHead()
        self.decoder = PromptableDecoder()
        self.image_pe = PositionEmbeddingRandom()

        self.init_pose = nn.Embedding(1, config.npose)
        self.init_camera = nn.Embedding(1, config.camera_ncam)
        nn.init.zeros_(self.init_camera.weight)

        init_dim = config.npose + config.camera_ncam + config.cond_dim
        self.init_to_token = nn.Linear(init_dim, config.decoder_dim)
        self.prev_to_token = nn.Linear(init_dim - config.cond_dim, config.decoder_dim)

        self.keypoint_embedding = nn.Embedding(config.n_keypoints, config.decoder_dim)
        self.keypoint_posemb_linear = MLP(
            in_dims=2,
            hidden_dims=config.decoder_dim,
            out_dims=config.decoder_dim,
            num_layers=2,
        )
        self.keypoint_feat_linear = nn.Linear(config.decoder_context_dim, config.decoder_dim)

        self.keypoint3d_embedding = nn.Embedding(config.n_keypoints, config.decoder_dim)
        self.keypoint3d_posemb_linear = MLP(
            in_dims=3,
            hidden_dims=config.decoder_dim,
            out_dims=config.decoder_dim,
            num_layers=2,
        )

    # ------------------------------------------------------------------
    # Token update callbacks
    # ------------------------------------------------------------------

    def _keypoint_token_update(
        self,
        kps_emb_start_idx: int,
        image_embeddings: BackboneOutput,
        token_embeddings: TokenSequence,
        token_augment: TokenSequence,
        pose_output: IteratedPoseOutput,
        layer_idx: int,
        affine_trans: AffineTransTensor,
        img_size: CroppedSizeTensor,
    ) -> tuple[TokenSequence, TokenSequence, IteratedPoseOutput, int]:
        """
        Refresh 2D keypoint tokens using intermediate pose predictions.

        Parameters
        ----------
        kps_emb_start_idx : int
            Start index of keypoint tokens in the token sequence.
        image_embeddings : BackboneOutput
            Ray-conditioned feature map, shape (B, C, Hp, Wp).
        token_embeddings : TokenSequence
            Current token sequence, shape (B, N, DEC_DIM).
        token_augment : TokenSequence
            Current token positional encodings, shape (B, N, DEC_DIM).
        pose_output : IteratedPoseOutput
            Intermediate pose output from the current decoder layer.
        layer_idx : int
            Current decoder layer index.
        affine_trans : AffineTransTensor
            Affine transforms from full image to crop, shape (B, 2, 3).
        img_size : CropSizeTensor
            Crop resolution (width, height) in pixels, shape (B, 2).

        Returns
        -------
        tuple
            Updated token_embeddings, token_augment, pose_output, layer_idx.
        """
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()
        num_kps = self.keypoint_embedding.weight.shape[0]

        kps = full_to_crop(
            pose_output.proj_out.pred_keypoints_2d,
            affine_trans,
            img_size,
        )
        depth = pose_output.proj_out.pred_keypoints_2d_depth

        kps_dt = self.keypoint_posemb_linear.layers[0][0].weight.dtype
        kps_m = kps.to(kps_dt)
        kps_01 = kps_m + 0.5
        invalid = (
            (kps_01[:, :, 0] < 0)
            | (kps_01[:, :, 0] > 1)
            | (kps_01[:, :, 1] < 0)
            | (kps_01[:, :, 1] > 1)
            | (depth.to(kps_dt) < 1e-5)
        )

        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_kps] = (
            self.keypoint_posemb_linear(kps_m) * (~invalid[:, :, None])
        )

        sample_pts = kps_m * 2
        sample_pts[:, :, 0] = sample_pts[:, :, 0] / 12 * 16

        feats = (
            F.grid_sample(
                image_embeddings,
                sample_pts[:, :, None, :],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )
        feats = feats * (~invalid[:, :, None])
        token_embeddings[
            :, kps_emb_start_idx : kps_emb_start_idx + num_kps
        ] += self.keypoint_feat_linear(feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def _keypoint3d_token_update(
        self,
        kps3d_emb_start_idx: int,
        token_embeddings: TokenSequence,
        token_augment: TokenSequence,
        pose_output: IteratedPoseOutput,
        layer_idx: int,
    ) -> tuple[TokenSequence, TokenSequence, IteratedPoseOutput, int]:
        """
        Refresh 3D keypoint tokens using pelvis-centred coordinates.

        Parameters
        ----------
        kps3d_emb_start_idx : int
            Start index of 3D keypoint tokens in the token sequence.
        token_embeddings : TokenSequence
            Current token sequence, shape (B, N, DEC_DIM).
        token_augment : TokenSequence
            Current token positional encodings, shape (B, N, DEC_DIM).
        pose_output : IteratedPoseOutput
            Intermediate pose output from the current decoder layer.
        layer_idx : int
            Current decoder layer index.

        Returns
        -------
        tuple
            Updated token_embeddings, token_augment, pose_output, layer_idx.
        """
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_kps3d = self.keypoint3d_embedding.weight.shape[0]
        kps3d: Keypoints3DTensor = pose_output.mhr_out.pred_keypoints_3d.clone()
        pelvis = (
            kps3d[:, [config.pelvis_indices[0]]]
            + kps3d[:, [config.pelvis_indices[1]]]
        ) / 2
        kps3d = kps3d - pelvis

        token_augment = token_augment.clone()
        k3_dt = self.keypoint3d_posemb_linear.layers[0][0].weight.dtype
        token_augment[
            :, kps3d_emb_start_idx : kps3d_emb_start_idx + num_kps3d
        ] = self.keypoint3d_posemb_linear(kps3d.to(k3_dt))

        return token_embeddings, token_augment, pose_output, layer_idx

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: PoseEstimatorInput) -> PoseEstimatorOutput:
        """
        Run iterative pose estimation on a batch of image embeddings.

        Parameters
        ----------
        batch : PoseEstimatorInput
            Ray-conditioned image embeddings and geometric metadata.

        Returns
        -------
        PoseEstimatorOutput
            Full pose, mesh, and camera outputs for the batch.
        """
        n = batch.image_embeddings.shape[0]

        cam_int = batch.cam_int
        if cam_int.shape[0] == 1:
            cam_int = cam_int.expand(n, -1, -1)

        condition_info = get_decoder_condition(
            batch.bbox_center,
            batch.bbox_scale,
            cam_int,
        ).to(batch.image_embeddings.dtype)

        init_pose = self.init_pose.weight.expand(n, -1).unsqueeze(1)
        init_camera = self.init_camera.weight.expand(n, -1).unsqueeze(1)
        init_estimate = torch.cat([init_pose, init_camera], dim=-1)

        init_input = torch.cat([condition_info.view(n, 1, -1), init_estimate], dim=-1)
        token_embeddings: TokenSequence = self.init_to_token(init_input)
        num_pose_token = token_embeddings.shape[1]

        prev_embeddings = self.prev_to_token(init_estimate)
        token_embeddings = torch.cat([token_embeddings, prev_embeddings], dim=1)
        token_augment: TokenSequence = torch.zeros_like(token_embeddings)
        token_augment[:, [num_pose_token]] = prev_embeddings

        kps_emb_start_idx = token_embeddings.shape[1]
        token_embeddings = torch.cat(
            [token_embeddings, self.keypoint_embedding.weight[None].repeat(n, 1, 1)],
            dim=1,
        )
        token_augment = torch.cat(
            [token_augment, torch.zeros_like(token_embeddings[:, token_augment.shape[1]:])],
            dim=1,
        )

        kps3d_emb_start_idx = token_embeddings.shape[1]
        token_embeddings = torch.cat(
            [token_embeddings, self.keypoint3d_embedding.weight[None].repeat(n, 1, 1)],
            dim=1,
        )
        token_augment = torch.cat(
            [token_augment, torch.zeros_like(token_embeddings[:, token_augment.shape[1]:])],
            dim=1,
        )

        image_augment = self.image_pe(config.patch_grid).to(dtype=batch.image_embeddings.dtype)

        def _token_to_pose(
            tokens: TokenSequence,
            layer_idx: int = 0,
        ) -> IteratedPoseOutput:
            is_final = layer_idx == len(self.decoder.layers) - 1
            pose_token = tokens[:, 0]
            mhr_out = self.mhr_head(
                pose_token,
                init_estimate=init_pose.view(n, -1),
                full_output=is_final,
            )
            pred_cam: PerspectiveHeadOutput = self.camera_head(
                pose_token,
                init_estimate=init_camera.view(n, -1),
            )
            proj_out = perspective_projection(
                mhr_out.pred_keypoints_3d,
                pred_cam,
                batch.bbox_center,
                batch.bbox_scale[:, 0],
                cam_int,
                full_output=is_final,
            )
            return IteratedPoseOutput(
                mhr_out=mhr_out,
                pred_cam=pred_cam,
                proj_out=proj_out,
            )

        def _kp_update(
            token_embeddings: TokenSequence,
            token_augment: TokenSequence,
            pose_output: IteratedPoseOutput,
            layer_idx: int,
        ) -> tuple[TokenSequence, TokenSequence, IteratedPoseOutput, int]:
            token_embeddings, token_augment, pose_output, layer_idx = (
                self._keypoint_token_update(
                    kps_emb_start_idx,
                    batch.image_embeddings,
                    token_embeddings,
                    token_augment,
                    pose_output,
                    layer_idx,
                    affine_trans=batch.affine_trans,
                    img_size=batch.img_size,
                )
            )
            token_embeddings, token_augment, pose_output, layer_idx = (
                self._keypoint3d_token_update(
                    kps3d_emb_start_idx,
                    token_embeddings,
                    token_augment,
                    pose_output,
                    layer_idx,
                )
            )
            return token_embeddings, token_augment, pose_output, layer_idx

        _, all_pose_outputs = self.decoder(
            token_embeddings,
            batch.image_embeddings,
            token_to_pose_output_fn=_token_to_pose,
            keypoint_token_update_fn=_kp_update,
            token_augment=token_augment,
            image_augment=image_augment,
        )

        final: IteratedPoseOutput = all_pose_outputs[-1]
        mhr_out = final.mhr_out
        proj_out = final.proj_out

        verts_proj = perspective_projection(
            mhr_out.pred_vertices,
            final.pred_cam,
            batch.bbox_center,
            batch.bbox_scale[:, 0],
            cam_int,
            full_output=False,
        )

        return PoseEstimatorOutput(
            pred_keypoints_3d=mhr_out.pred_keypoints_3d,
            pred_keypoints_2d=proj_out.pred_keypoints_2d,
            pred_keypoints_2d_depth=proj_out.pred_keypoints_2d_depth,
            pred_vertices=mhr_out.pred_vertices,
            pred_keypoints_2d_verts=verts_proj.pred_keypoints_2d,
            pred_cam=final.pred_cam,
            pred_cam_t=proj_out.pred_cam_t,
            focal_length=proj_out.focal_length,
            pred_pose_raw=mhr_out.pred_pose_raw,
            global_rot=mhr_out.global_rot,
            body_pose=mhr_out.body_pose,
            shape=mhr_out.shape,
            scale=mhr_out.scale,
            hand=mhr_out.hand,
            faces=self.mhr_head.faces.detach().cpu(),
        )
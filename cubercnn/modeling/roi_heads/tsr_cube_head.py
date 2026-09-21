from detectron2.utils.registry import Registry
from typing import Dict, Tuple
from detectron2.layers import ShapeSpec
from torch import nn
import torch
import numpy as np
import fvcore.nn.weight_init as weight_init

from pytorch3d.transforms.rotation_conversions import _copysign
from pytorch3d.transforms import (
    rotation_6d_to_matrix, 
    euler_angles_to_matrix, 
    quaternion_to_matrix
)
from cubercnn.modeling.tsr import TSR, TSA
from cubercnn.modeling.roi_heads.cube_head import CubeHead

TSR_CUBE_HEAD_REGISTRY = Registry("TSR_CUBE_HEAD")

@TSR_CUBE_HEAD_REGISTRY.register()
class TSRCubeHead(CubeHead):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self.spatial_shape = (input_shape.channels, input_shape.height, input_shape.width)
        token_dim = cfg.MODEL.ROI_CUBE_HEAD.TSR.TOKEN_DIM
        hidden_dim = cfg.MODEL.ROI_CUBE_HEAD.TSR.HIDDEN_DIM
        temperature = cfg.MODEL.ROI_CUBE_HEAD.TSR.TEMPERATURE
        bottleneck_ratio = cfg.MODEL.ROI_CUBE_HEAD.TSR.BOTTLENECK_RATIO

        self.tsr = TSR(
            in_channels=input_shape.channels,
            hidden_dim=hidden_dim,
            token_dim=token_dim,
            tasks=("depth", "dim", "pose"),
            temperature=temperature)

        base_dim = self._output_size
        self.depth_adapter = TSA(token_dim=token_dim, base_dim=base_dim, bottleneck_ratio=bottleneck_ratio)
        self.dim_adapter = TSA(token_dim=token_dim, base_dim=base_dim, bottleneck_ratio=bottleneck_ratio)
        self.pose_adapter = TSA(token_dim=token_dim, base_dim=base_dim, bottleneck_ratio=bottleneck_ratio)

    def forward(
        self,
        x: torch.Tensor,
        return_tsr: bool = False
    ) -> Tuple:
        """
        Args:
            x (torch.Tensor): RoI feature [N, C, H, W]
            return_tsr (bool): Return TSR map
        """
        n = x.shape[0]

        if x.dim() == 2:
            x_spatial = x.view(n, *self.spatial_shape)  # [N, C, H, W]
            x_flat = x                                  # [N, 12544]
        else:
            x_spatial = x                               # [N, C, H, W]
            x_flat = x.flatten(1)                       # [N, 12544]

        # 1. 空間 4D 特徵圖丟給 TSR 計算 Token
        need_maps = self.training or return_tsr
        tsr_out = self.tsr(x_spatial, mode="roi", return_maps=need_maps)

        # 2. 原始 2D/展平特徵圖丟給 原生的 feature_generator (內部是 nn.Linear)
        if self.shared_fc:
            feat_shared = self.feature_generator(x_flat)
            feat_xy = feat_shared
            feat_dims = feat_shared
            feat_pose = feat_shared
            feat_z = feat_shared
            if self.use_conf:
                feat_conf = feat_shared
        else:
            feat_xy = self.feature_generator_XY(x_flat)
            feat_dims = self.feature_generator_dims(x_flat)
            feat_pose = self.feature_generator_pose(x_flat)
            feat_z = self.feature_generator_Z(x_flat)
            if self.use_conf:
                feat_conf = self.feature_generator_conf(x_flat)

        # 3. TSA 特徵注入調製
        feat_z = self.depth_adapter(feat_z, tsr_out["depth_token"])
        feat_dims = self.dim_adapter(feat_dims, tsr_out["dim_token"])
        feat_pose = self.pose_adapter(feat_pose, tsr_out["pose_token"])

        # 4. 後續預測頭計算（保持原樣）...
        box_2d_deltas = self.bbox_3D_center_deltas(feat_xy)
        box_dims = self.bbox_3D_dims(feat_dims)
        box_pose = self.bbox_3D_pose(feat_pose)
        box_z = self.bbox_3D_center_depth(feat_z)

        box_uncert = None
        if self.use_conf:
            box_uncert = self.bbox_3D_uncertainty(feat_conf).clip(0.01)

        if self.pose_type == '6d':
            box_pose = rotation_6d_to_matrix(box_pose.view(-1, 6))
        elif self.pose_type == 'quaternion':
            quats = box_pose.view(-1, 4)
            quats_scales = (quats * quats).sum(1)
            quats = quats / _copysign(torch.sqrt(quats_scales), quats[:, 0])[:, None]
            box_pose = quaternion_to_matrix(quats)
        elif self.pose_type == 'euler':
            box_pose = euler_angles_to_matrix(box_pose.view(-1, 3), 'XYZ')

        box_2d_deltas = box_2d_deltas.view(n, self.num_classes, 2)
        box_dims = box_dims.view(n, self.num_classes, 3)
        box_pose = box_pose.view(n, self.num_classes, 3, 3)

        if self.cluster_bins > 1:
            box_z = box_z.view(n, self.cluster_bins, self.num_classes, -1)
        else:
            box_z = box_z.view(n, self.num_classes, -1)

        cube_preds = (box_2d_deltas, box_z, box_dims, box_pose, box_uncert)

        if return_tsr:
            return cube_preds, tsr_out
        return cube_preds


def build_tsr_cube_head(cfg, input_shape: Dict[str, ShapeSpec]):
    name = cfg.MODEL.ROI_CUBE_HEAD.NAME
    return TSR_CUBE_HEAD_REGISTRY.get(name)(cfg, input_shape)
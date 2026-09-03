# from typing import Dict, List, Tuple, Optional
# import torch
# import numpy as np

# from detectron2.config import configurable
# from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN
# from detectron2.modeling.backbone import build_backbone
# from detectron2.structures import Instances
# from detectron2.utils.events import get_event_storage
# from detectron2.utils.logger import _log_api_usage
# from detectron2.data.detection_utils import convert_image_to_rgb
# from detectron2.data import MetadataCatalog
# from detectron2.utils.visualizer import Visualizer
# from detectron2.layers import batched_nms

# from cubercnn.modeling.dense_head import build_dense_cube_head
# from cubercnn import util, vis


# @META_ARCH_REGISTRY.register()
# class DenseCube3D(torch.nn.Module):
#     """
#     Single-Stage 3D Object Detector architecture (Counterpart of 2-stage RCNN3D).
#     Directly connects Backbone -> DenseCubeHead without RPN or RoIAlign.
#     """

#     @configurable
#     def __init__(
#         self,
#         *,
#         backbone: torch.nn.Module,
#         head: torch.nn.Module,
#         head_in_features: List[str],
#         input_format: str = "BGR",
#         vis_period: int = 0,
#         pixel_mean: List[float],
#         pixel_std: List[float],
#     ):
#         super().__init__()
#         self.backbone = backbone
#         self.head = head
#         self.head_in_features = head_in_features
#         self.input_format = input_format
#         self.vis_period = vis_period

#         self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
#         self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

#     @classmethod
#     def from_config(cls, cfg, priors: Optional[dict] = None):
#         backbone = build_backbone(cfg)
#         head = build_dense_cube_head(cfg, backbone.output_shape(), priors=priors)
#         return {
#             "backbone": backbone,
#             "head": head,
#             "head_in_features": cfg.MODEL.DENSE_HEAD.IN_FEATURES,
#             "input_format": cfg.INPUT.FORMAT,
#             "vis_period": cfg.VIS_PERIOD,
#             "pixel_mean": cfg.MODEL.PIXEL_MEAN,
#             "pixel_std": cfg.MODEL.PIXEL_STD,
#         }

#     @property
#     def device(self) -> torch.device:
#         return self.pixel_mean.device

#     def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         Normalize, pad and batch the input images.
#         """
#         from detectron2.structures import ImageList
#         images = [x["image"].to(self.device) for x in batched_inputs]
#         images = [(x - self.pixel_mean) / self.pixel_std for x in images]
#         images = ImageList.from_tensors(images, self.backbone.size_divisibility)
#         return images

#     def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         if not self.training:
#             return self.inference(batched_inputs)

#         images = self.preprocess_image(batched_inputs)

#         # Ratio between original image height and processed unpadded height
#         im_scales_ratio = [
#             float(info["height"]) / float(image_size[0])
#             for info, image_size in zip(batched_inputs, images.image_sizes)
#         ]
#         Ks = [torch.as_tensor(info["K"], dtype=torch.float32) for info in batched_inputs]

#         # Current unpadded image dimensions (H, W) per sample
#         im_current_dims = [(float(h), float(w)) for (h, w) in images.image_sizes]

#         gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

#         features = self.backbone(images.tensor)
#         outputs = self.head(features)

#         losses = self.head.losses(
#             outputs, features, gt_instances, Ks, im_scales_ratio, im_current_dims
#         )

#         if self.vis_period > 0:
#             storage = get_event_storage()
#             if storage.iter % self.vis_period == 0 and storage.iter > 0:
#                 self.visualize_training(batched_inputs, images, outputs)

#         return losses

#     @torch.no_grad()
#     def inference(
#         self,
#         batched_inputs: List[Dict[str, torch.Tensor]],
#         do_postprocess: bool = True,
#     ):
#         assert not self.training

#         images = self.preprocess_image(batched_inputs)
#         im_scales_ratio = [
#             float(info["height"]) / float(image_size[0])
#             for info, image_size in zip(batched_inputs, images.image_sizes)
#         ]
#         Ks = [torch.as_tensor(info["K"], dtype=torch.float32) for info in batched_inputs]
#         im_current_dims = [(float(h), float(w)) for (h, w) in images.image_sizes]

#         features = self.backbone(images.tensor)
#         outputs = self.head(features)

#         results = self.head.inference(
#             outputs, features, Ks, images.image_sizes, im_scales_ratio, im_current_dims
#         )

#         if do_postprocess:
#             assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
#             return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
#         return results

#     def visualize_training(self, batched_inputs, images, outputs):
#         """
#         Visualizes 2D bounding boxes and 3D cuboids during training.
#         """
#         storage = get_event_storage()
#         max_vis_prop = 20

#         if not hasattr(self, "thing_classes"):
#             self.thing_classes = MetadataCatalog.get("omni3d_model").thing_classes
#             self.num_classes = len(self.thing_classes)

#         im_scales_ratio = [
#             float(info["height"]) / float(image_size[0])
#             for info, image_size in zip(batched_inputs, images.image_sizes)
#         ]
#         Ks = [torch.as_tensor(info["K"], dtype=torch.float32) for info in batched_inputs]
#         im_current_dims = [(float(h), float(w)) for (h, w) in images.image_sizes]

#         with torch.no_grad():
#             results = self.head.inference(
#                 outputs, self.backbone(images.tensor), Ks, images.image_sizes, im_scales_ratio, im_current_dims
#             )

#         for input_info, instances_i in zip(batched_inputs, results):
#             img = input_info["image"]
#             img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
#             img_3DGT = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]])
#             img_3DPR = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]])

#             v_gt = Visualizer(img, None)
#             v_gt = v_gt.overlay_instances(boxes=input_info["instances"].gt_boxes)
#             anno_img = v_gt.get_image()

#             if len(instances_i) == 0:
#                 break

#             keep = batched_nms(
#                 instances_i.pred_boxes.tensor,
#                 instances_i.scores,
#                 torch.zeros(len(instances_i.scores), dtype=torch.long, device=instances_i.scores.device),
#                 self.head.test_nms_thresh,
#             )[:max_vis_prop]

#             v_pred = Visualizer(img, None)
#             v_pred = v_pred.overlay_instances(boxes=instances_i.pred_boxes[keep].tensor.cpu().numpy())
#             prop_img = v_pred.get_image()

#             vis_img_rpn = np.concatenate((anno_img, prop_img), axis=1).transpose(2, 0, 1)
#             storage.put_image("Left: GT 2D boxes; Right: Predicted 2D boxes", vis_img_rpn)

#             K = torch.tensor(input_info["K"], device=self.device)
#             scale = input_info["height"] / img.shape[0]
#             K_scaled = torch.tensor(
#                 [[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1.0]],
#                 dtype=torch.float32, device=self.device,
#             ) @ K

#             gts_per_image = input_info["instances"]
#             gt_classes = gts_per_image.gt_classes
#             fg = (gt_classes != -1) & (gt_classes < self.num_classes)
#             gt_classes = gt_classes[fg]
#             gt_class_names = [self.thing_classes[c] for c in gt_classes]
#             gt_poses = gts_per_image.gt_poses[fg]
#             gt_boxes3D = gts_per_image.gt_boxes3D[fg]

#             fx, sx = (v.item() / scale for v in K[0, [0, 2]])
#             fy, sy = (v.item() / scale for v in K[1, [1, 2]])
#             gt_z = gt_boxes3D[:, 2]
#             gt_x3D = gt_z * (gt_boxes3D[:, 0] - sx) / fx
#             gt_y3D = gt_z * (gt_boxes3D[:, 1] - sy) / fy
#             gt_center_3D = torch.stack((gt_x3D, gt_y3D, gt_z)).T
#             gt_boxes3D_XYZ_WHL = torch.cat((gt_center_3D, gt_boxes3D[:, 3:6]), dim=1)
#             gt_colors = torch.tensor(
#                 [util.get_color(i) for i in range(len(gt_boxes3D_XYZ_WHL))], device=self.device
#             ) / 255.0
#             gt_meshes = util.mesh_cuboid(gt_boxes3D_XYZ_WHL, gt_poses, gt_colors)

#             pred_xyzwhl = torch.cat(
#                 (instances_i.pred_center_cam[keep], instances_i.pred_dimensions[keep]), dim=1
#             )
#             pred_pose = instances_i.pred_pose[keep]
#             pred_colors = torch.tensor(
#                 [util.get_color(i) for i in range(len(keep))], device=self.device
#             ) / 255.0
#             pred_classes = instances_i.pred_classes[keep]
#             pred_scores = instances_i.scores[keep]
#             pred_class_names = [
#                 "{} {:.2f}".format(self.thing_classes[c], s) for c, s in zip(pred_classes, pred_scores)
#             ]
#             pred_meshes = util.mesh_cuboid(pred_xyzwhl, pred_pose, pred_colors)

#             pred_meshes = [pred_meshes.__getitem__(i).detach() for i in range(len(pred_meshes))]
#             gt_meshes = [gt_meshes.__getitem__(i) for i in range(len(gt_meshes))]

#             img_3DPR = vis.draw_scene_view(
#                 img_3DPR, K_scaled.cpu().numpy(), pred_meshes, text=pred_class_names,
#                 mode="front", blend_weight=0.0, blend_weight_overlay=0.85,
#             )
#             img_3DGT = vis.draw_scene_view(
#                 img_3DGT, K_scaled.cpu().numpy(), gt_meshes, text=gt_class_names,
#                 mode="front", blend_weight=0.0, blend_weight_overlay=0.85,
#             )
#             vis_img_3d = np.concatenate((img_3DGT, img_3DPR), axis=1)[:, :, [2, 1, 0]]
#             vis_img_3d = vis_img_3d.astype(np.uint8).transpose(2, 0, 1)
#             storage.put_image("Left: GT 3D cuboids; Right: Predicted 3D cuboids", vis_img_3d)
#             break


# def build_dense_model(cfg, priors: Optional[dict] = None):
#     model = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)(cfg, priors=priors)
#     model.to(torch.device(cfg.MODEL.DEVICE))
#     _log_api_usage("modeling.meta_arch." + cfg.MODEL.META_ARCHITECTURE)
#     return model

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

from detectron2.config import configurable
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import _log_api_usage
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.layers import batched_nms

from cubercnn.modeling.dense_head import build_dense_cube_head, build_cascade_dense_cube_head
from cubercnn import util, vis


@META_ARCH_REGISTRY.register()
class DenseCube3D(torch.nn.Module):
    """
    單階段 3D 物件檢測架構 (對應原版 2-Stage RCNN3D)。
    Backbone -> DenseCubeHead 直接端到端輸出 2D 與 3D 方體預測。
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        head_in_features: List[str],
        input_format: str = "BGR",
        vis_period: int = 0,
        pixel_mean: List[float],
        pixel_std: List[float],
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features
        self.input_format = input_format
        self.vis_period = vis_period

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg, priors: Optional[dict] = None):
        backbone = build_backbone(cfg)
        head = build_cascade_dense_cube_head(cfg, backbone.output_shape(), priors=priors)
        return {
            "backbone": backbone,
            "head": head,
            "head_in_features": cfg.MODEL.DENSE_HEAD.IN_FEATURES,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        影像正規化、Padding 與 Batch 打包。
        """
        from detectron2.structures import ImageList
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        # 原始高度與當前縮放高度之比例 (用於還原投影座標)
        im_scales_ratio = [
            float(info["height"]) / float(image_size[0])
            for info, image_size in zip(batched_inputs, images.image_sizes)
        ]
        Ks = [torch.as_tensor(info["K"], dtype=torch.float32) for info in batched_inputs]
        im_current_dims = [(float(h), float(w)) for (h, w) in images.image_sizes]

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)
        outputs = self.head(features)

        losses = self.head.losses(
            outputs, features, gt_instances, Ks, im_scales_ratio, im_current_dims
        )

        # -------------------------------------------------------------
        # 視覺化邏輯：當達指定 iteration 時將 2D/3D GT 與預測寫入 TensorBoard
        # -------------------------------------------------------------
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0 and storage.iter > 0:
                self.visualize_training(batched_inputs, images, outputs)

        return losses

    @torch.no_grad()
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        im_scales_ratio = [
            float(info["height"]) / float(image_size[0])
            for info, image_size in zip(batched_inputs, images.image_sizes)
        ]
        Ks = [torch.as_tensor(info["K"], dtype=torch.float32) for info in batched_inputs]
        im_current_dims = [(float(h), float(w)) for (h, w) in images.image_sizes]

        features = self.backbone(images.tensor)
        outputs = self.head(features)

        results = self.head.inference(
            outputs, features, Ks, images.image_sizes, im_scales_ratio, im_current_dims
        )

        if do_postprocess:
            assert not torch.jit.is_scripting(), "不支援 TorchScript postprocess"
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    # -----------------------------------------------------------------
    # TensorBoard 訓練視覺化函式 (完全比照原版 RCNN3D 呈現格式)
    # -----------------------------------------------------------------
    def visualize_training(self, batched_inputs, images, outputs):
        storage = get_event_storage()
        max_vis_prop = 20

        if not hasattr(self, "thing_classes"):
            self.thing_classes = MetadataCatalog.get("omni3d_model").thing_classes
            self.num_classes = len(self.thing_classes)

        im_scales_ratio = [
            float(info["height"]) / float(image_size[0])
            for info, image_size in zip(batched_inputs, images.image_sizes)
        ]
        Ks = [torch.as_tensor(info["K"], dtype=torch.float32) for info in batched_inputs]
        im_current_dims = [(float(h), float(w)) for (h, w) in images.image_sizes]

        # 針對當前 batch 執行一次推論解碼以繪製預測框
        with torch.no_grad():
            results = self.head.inference(
                outputs, self.backbone(images.tensor), Ks, images.image_sizes, im_scales_ratio, im_current_dims
            )

        for input_info, instances_i in zip(batched_inputs, results):
            img = input_info["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            img_3DGT = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]]) # BGR
            img_3DPR = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]]) # BGR

            # ---------------------------------------------------------
            # 1. 繪製 2D Ground Truth 與 2D 預測框 (Left: GT, Right: Pred)
            # ---------------------------------------------------------
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input_info["instances"].gt_boxes)
            anno_img = v_gt.get_image()

            if len(instances_i) == 0:
                break

            keep = batched_nms(
                instances_i.pred_boxes.tensor,
                instances_i.scores,
                torch.zeros(len(instances_i.scores), dtype=torch.long, device=instances_i.scores.device),
                self.head.test_nms_thresh,
            )[:max_vis_prop]

            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(boxes=instances_i.pred_boxes[keep].tensor.cpu().numpy())
            prop_img = v_pred.get_image()

            vis_img_2d = np.concatenate((anno_img, prop_img), axis=1)
            vis_img_2d = vis_img_2d.transpose(2, 0, 1) # HWC -> CHW
            storage.put_image("Left: GT 2D boxes; Right: Predicted 2D boxes", vis_img_2d)

            # ---------------------------------------------------------
            # 2. 繪製 3D Ground Truth 與 3D 預測立方體 (Left: GT, Right: Pred)
            # ---------------------------------------------------------
            K = torch.tensor(input_info["K"], device=self.device)
            scale = input_info["height"] / img.shape[0]
            K_scaled = torch.tensor(
                [[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1.0]],
                dtype=torch.float32, device=self.device,
            ) @ K

            gts_per_image = input_info["instances"]
            gt_classes = gts_per_image.gt_classes
            fg = (gt_classes != -1) & (gt_classes < self.num_classes)
            gt_classes = gt_classes[fg]
            gt_class_names = [self.thing_classes[c] for c in gt_classes]
            gt_poses = gts_per_image.gt_poses[fg]
            gt_boxes3D = gts_per_image.gt_boxes3D[fg]

            # 2D 投影逆變換計算 GT 3D 中心點
            fx, sx = (v.item() / scale for v in K[0, [0, 2]])
            fy, sy = (v.item() / scale for v in K[1, [1, 2]])
            gt_z = gt_boxes3D[:, 2]
            gt_x3D = gt_z * (gt_boxes3D[:, 0] - sx) / fx
            gt_y3D = gt_z * (gt_boxes3D[:, 1] - sy) / fy
            gt_center_3D = torch.stack((gt_x3D, gt_y3D, gt_z)).T
            gt_boxes3D_XYZ_WHL = torch.cat((gt_center_3D, gt_boxes3D[:, 3:6]), dim=1)
            gt_colors = torch.tensor(
                [util.get_color(i) for i in range(len(gt_boxes3D_XYZ_WHL))], device=self.device
            ) / 255.0
            gt_meshes = util.mesh_cuboid(gt_boxes3D_XYZ_WHL, gt_poses, gt_colors)

            # 預測方體 Mesh 構建
            pred_xyzwhl = torch.cat(
                (instances_i.pred_center_cam[keep], instances_i.pred_dimensions[keep]), dim=1
            )
            pred_pose = instances_i.pred_pose[keep]
            pred_colors = torch.tensor(
                [util.get_color(i) for i in range(len(keep))], device=self.device
            ) / 255.0
            pred_classes = instances_i.pred_classes[keep]
            pred_scores = instances_i.scores[keep]
            pred_class_names = [
                "{} {:.2f}".format(self.thing_classes[c], s) for c, s in zip(pred_classes, pred_scores)
            ]
            pred_meshes = util.mesh_cuboid(pred_xyzwhl, pred_pose, pred_colors)

            pred_meshes = [pred_meshes.__getitem__(i).detach() for i in range(len(pred_meshes))]
            gt_meshes = [gt_meshes.__getitem__(i) for i in range(len(gt_meshes))]

            # 渲染前視角 3D Cuboids
            img_3DPR = vis.draw_scene_view(
                img_3DPR, K_scaled.cpu().numpy(), pred_meshes, text=pred_class_names,
                mode="front", blend_weight=0.0, blend_weight_overlay=0.85,
            )
            img_3DGT = vis.draw_scene_view(
                img_3DGT, K_scaled.cpu().numpy(), gt_meshes, text=gt_class_names,
                mode="front", blend_weight=0.0, blend_weight_overlay=0.85,
            )

            # 左右拼接並轉為 RGB 傳送至 TensorBoard
            vis_img_3d = np.concatenate((img_3DGT, img_3DPR), axis=1)[:, :, [2, 1, 0]]
            vis_img_3d = vis_img_3d.astype(np.uint8).transpose(2, 0, 1)
            storage.put_image("Left: GT 3D cuboids; Right: Predicted 3D cuboids", vis_img_3d)

            break  # 每個 batch 僅視覺化第一張圖片以節省顯存與時間


def build_dense_model(cfg, priors: Optional[dict] = None):
    model = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)(cfg, priors=priors)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + cfg.MODEL.META_ARCHITECTURE)
    return model

from typing import Dict, Iterable, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils import registry

from ultralytics.nn.modules import LightConv, Conv


# TODO Merge into detectron2 arch
class TSR(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 128,
        token_dim: int = 256,
        tasks: Sequence[str] = ("depth", "dim", "pose"),
        temperature: float = 1.0,
        dense_gate_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        self.tasks = tuple(tasks)
        self.temperature = float(temperature)

        self.encoder = LightConv(self.in_channels, self.hidden_dim, k=3, act=nn.SiLU())

        # evidence head without act function
        self.evidence_head = nn.Conv2d(
            self.hidden_dim,
            len(self.tasks),
            kernel_size=1,
            bias=True,
        )

        self.task_projections = nn.ModuleDict({
            task: Conv(self.hidden_dim, self.token_dim, k=1, act=True)
            for task in self.tasks
        })

        self._initialize_parameters(dense_gate_bias)

    def _initialize_parameters(self, dense_gate_bias: float) -> None:
        nn.init.normal_(self.evidence_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.evidence_head.bias, dense_gate_bias)

    def _task_index(self, task: str) -> int:
        return self.tasks.index(task)

    def _roi_evidence_map(self, task_logits: torch.Tensor) -> torch.Tensor:
        B, _, H, W = task_logits.shape
        logits_flat = task_logits.flatten(start_dim=2)
        evidence = F.softmax(logits_flat.float() / self.temperature, dim=-1).to(dtype=task_logits.dtype)
        return evidence.reshape(B, 1, H, W)

    @staticmethod
    def _weighted_pool(value_map: torch.Tensor, evidence_map: torch.Tensor) -> torch.Tensor:
        if value_map.shape[0] != evidence_map.shape[0]:
            raise ValueError("Batch sizes do not match.")
        if value_map.shape[-2:] != evidence_map.shape[-2:]:
            raise ValueError("Spatial dimensions do not match.")
        return torch.sum(value_map * evidence_map, dim=(-2, -1))

    def _forward_roi(
        self,
        feat: torch.Tensor,
        logits: torch.Tensor,
        selected_tasks: Tuple[str, ...],
        return_maps: bool,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        task_maps = [] if return_maps else None

        for task in selected_tasks:
            task_idx = self._task_index(task)
            task_logits = logits[:, task_idx:task_idx + 1]
            evidence_map = self._roi_evidence_map(task_logits)

            value_map = self.task_projections[task](feat)
            token = self._weighted_pool(value_map=value_map, evidence_map=evidence_map)

            outputs[f"{task}_token"] = token
            if return_maps:
                outputs[f"{task}_map"] = evidence_map
                task_maps.append(evidence_map)

        # Return
        if return_maps and task_maps:
            outputs["evidence_maps"] = torch.stack(task_maps, dim=1)

        return outputs

    def _forward_dense(
        self,
        feat: torch.Tensor,
        logits: torch.Tensor,
        selected_tasks: Tuple[str, ...],
        return_maps: bool,
        collect_dense: bool,
        dense_consumer=None,
    ) -> Dict[str, torch.Tensor]:
        if not collect_dense and dense_consumer is None:
            raise ValueError("dense_consumer must be provided when collect_dense=False.")

        outputs: Dict[str, torch.Tensor] = {}
        task_maps = [] if return_maps else None

        for task in selected_tasks:
            task_idx = self._task_index(task)
            task_logits = logits[:, task_idx:task_idx + 1]

            gate = torch.sigmoid(task_logits.float() / self.temperature).to(dtype=task_logits.dtype)
            value_map = self.task_projections[task](feat)
            routed_feature = value_map * gate

            if collect_dense:
                outputs[f"{task}_feat"] = routed_feature
            else:
                dense_consumer(task, routed_feature, gate)

            if return_maps:
                outputs[f"{task}_map"] = gate
                task_maps.append(gate)

        # Return [B, T, 1, H, W]
        if return_maps and task_maps:
            outputs["evidence_maps"] = torch.stack(task_maps, dim=1)

        return outputs

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "roi",
        task_subset: Optional[Iterable[str]] = None,
        return_maps: bool = True,
        collect_dense: bool = True,
        dense_consumer=None,
    ) -> Dict[str, torch.Tensor]:
        selected_tasks = tuple(task_subset) if task_subset is not None else self.tasks

        feat = self.encoder(x)
        logits = self.evidence_head(feat)

        if mode == "roi":
            return self._forward_roi(
                feat=feat,
                logits=logits,
                selected_tasks=selected_tasks,
                return_maps=return_maps,
            )
        elif mode == "dense":
            return self._forward_dense(
                feat=feat,
                logits=logits,
                selected_tasks=selected_tasks,
                return_maps=return_maps,
                collect_dense=collect_dense,
                dense_consumer=dense_consumer,
            )
        else:
            raise ValueError(f"Unsupported mode={mode!r}.")


class TSA(nn.Module):
    """
    TSR adapter
    """
    def __init__(
            self,
            token_dim: int = 256,
            base_dim: int = 1024,
            bottleneck_ratio: float = 0.5,) -> None:
        super().__init__()
        mid_dim = max(16, int(token_dim * bottleneck_ratio))
        self.adapter = nn.Sequential(
            Conv(token_dim, mid_dim, k=1, act=True),
            Conv(mid_dim, base_dim, k=1, act=False),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
            self,
            base_feature: torch.Tensor,
            token: torch.Tensor,) -> torch.Tensor:
        """
        Args:
            base_feature: [N, base_dim] or [B, base_dim, H, W]
            token:        [N, token_dim] or [B, token_dim, H, W]
        """
        is_roi = (base_feature.dim() == 2)

        if is_roi:
            # [N, C] -> [N, C, 1, 1]
            res = self.adapter(token.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            res = self.adapter(token)

        return base_feature + self.alpha * res
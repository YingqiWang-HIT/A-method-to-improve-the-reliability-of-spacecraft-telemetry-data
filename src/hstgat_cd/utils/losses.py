from __future__ import annotations
import torch


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = ((pred - target) ** 2) * mask
    return diff2.sum() / mask.sum().clamp_min(1.0)


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def subsystem_soft_sharing_loss(model, subsystem_indices: list[list[int]]) -> torch.Tensor:
    losses = []
    if not hasattr(model.temporal_encoder, 'node_linears'):
        return torch.tensor(0.0, device=next(model.parameters()).device)
    weights = model.temporal_encoder.node_linears.weight  # [num_nodes, hidden]
    for indices in subsystem_indices:
        if len(indices) < 2:
            continue
        sub_w = weights[indices]
        center = sub_w.mean(dim=0, keepdim=True)
        losses.append(((sub_w - center) ** 2).mean())
    if not losses:
        return torch.tensor(0.0, device=weights.device)
    return torch.stack(losses).mean()

"""SSRL reconstruction, RBF topology and sliced-Wasserstein objectives."""
import torch
from torch import nn
from torch.nn import functional as F


def topology_loss(rgb, xyz, tau=1.0):
    if tau <= 0:
        raise ValueError("topology temperature must be positive")    rgb = rgb.flatten(2).transpose(1, 2).float()
    xyz = xyz.flatten(2).transpose(1, 2).float()

    def graph(x):
        distances = (
            x.square().sum(-1, keepdim=True)
            + x.square().sum(-1).unsqueeze(1)
            - 2 * x @ x.transpose(1, 2)
        )
        return torch.exp(-distances.clamp_min(0) / tau)    return (graph(rgb) - graph(xyz)).square().sum(dim=(1, 2)).mean()


def sliced_wasserstein(current, previous, projections=32, max_tokens=512):
    """Deterministic sliced W2 approximation; reference branch is detached."""
    if current.shape != previous.shape:
        raise ValueError(
            "teacher and student representations must have identical shapes"
        )
    if projections < 1 or max_tokens < 1:
        raise ValueError("projections and max_tokens must be positive")
    channels = current.shape[1]
    x = current.movedim(1, -1).reshape(-1, channels).float()
    y = previous.detach().movedim(1, -1).reshape(-1, channels).float()
    indices = torch.linspace(
        0, x.shape[0] - 1, min(x.shape[0], max_tokens), device=x.device
    ).long()
    x, y = x[indices], y[indices]
    generator = torch.Generator(device="cpu").manual_seed(133)
    directions = F.normalize(
        torch.randn(channels, projections, generator=generator), dim=0
    ).to(x.device)
    squared = (
        ((x @ directions).sort(dim=0).values - (y @ directions).sort(dim=0).values)
        .square()
        .mean()
    )    return torch.sqrt(squared + 1e-12) - 1e-6


class SSRLLoss(nn.Module):
    def __init__(
        self,
        weight=1.0,
        topology_weight=0.01,
        drift_weight=0.1,
        tau=1.0,
        projections=32,
        max_tokens=512,
    ):
        super().__init__()
        if topology_weight < 0 or drift_weight < 0 or tau <= 0:
            raise ValueError("loss weights must be nonnegative and tau positive")
        self.weight = weight
        self.topology_weight, self.drift_weight = topology_weight, drift_weight
        self.tau, self.projections, self.max_tokens = tau, projections, max_tokens

    def components(self, outputs):
        residual = outputs["feature_rec_final"] - outputs["feature_target"].detach()
        reconstruction = residual.square().sum(dim=1).mean()
        topology = topology_loss(
            outputs["topology_rgb"], outputs["topology_xyz"], self.tau
        )
        drift = reconstruction.new_zeros(())
        if "previous_representation" in outputs and self.drift_weight:
            drift = sliced_wasserstein(
                outputs["representation"],
                outputs["previous_representation"],
                self.projections,
                self.max_tokens,
            )
        return reconstruction, topology, drift

    def forward(self, outputs):
        reconstruction, topology, drift = self.components(outputs)
        return (
            reconstruction + self.topology_weight * topology + self.drift_weight * drift
        )

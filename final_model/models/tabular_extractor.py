import torch
import torch.nn as nn


class TabularFeatureExtractor(nn.Module):
    """
    Higgs analogue of CNNFeatureExtractor: a small MLP standing in for the
    ResNet backbone, since HIGGS features are already tabular (no images).
    Keeps the same forward()/embed()/to_angles interface so the existing
    meta-learning loops (meta/qmaml_loops.py, meta/qmaml_paper_loops.py,
    meta/reptile_loops.py) work unmodified regardless of data modality.
    """
    def __init__(self, input_dim: int, output_dim: int, num_qubits: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.ReLU(),
        )
        self.to_angles = nn.Linear(output_dim, num_qubits)
        self.register_buffer("_mean", torch.zeros(input_dim))
        self.register_buffer("_std", torch.ones(input_dim))

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._mean.copy_(mean)
        self._std.copy_(std.clamp_min(1e-6))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self._mean) / self._std

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(self._norm(x))
        return self.to_angles(feat)


__all__ = ["TabularFeatureExtractor"]

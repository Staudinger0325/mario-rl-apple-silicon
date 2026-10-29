"""The small actor-critic network used by Viet Nguyen's pretrained PPO agent."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ActorCritic(nn.Module):
    """Keep layer names and activations identical to the upstream checkpoint."""

    def __init__(self, num_inputs: int = 4, num_actions: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # The upstream model intentionally has no activation after this layer.
        x = self.linear(x.reshape(x.shape[0], -1))
        return self.actor_linear(x), self.critic_linear(x)


def select_device(name: str = "auto") -> torch.device:
    """Choose a supported accelerator; explicit unavailable choices fail clearly."""
    if name == "auto":
        if torch.backends.mps.is_available():
            name = "mps"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Use --device cpu or an ARM64 PyTorch installation.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. On Apple Silicon use --device mps or cpu.")
    if device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported device: {name}")
    return device


def load_model(path: str | Path, device: str | torch.device = "cpu") -> ActorCritic:
    """Load tensor-only weights without permitting arbitrary pickle execution."""
    selected = select_device(str(device))
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a state_dict or a checkpoint containing 'model'.")
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint 'model' must be a state_dict.")
    model = ActorCritic()
    model.load_state_dict(state_dict, strict=True)
    return model.to(selected).eval()

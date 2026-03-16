"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


# DONE: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        chunk_size=16,
        d_model=256,
        depth=3,
        dropout=0.0,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
        )
        layers: list[nn.Module] = []
        input_dim = state_dim
        for _ in range(depth):
            layers.append(nn.Linear(input_dim, d_model))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = d_model
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        return self.mlp(state).view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(state)


# DONE: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self,
        state_dim,
        action_dim,
        chunk_size=16,
        d_model=256,
        depth=3,
        dropout=0.1,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
        )
        feature_dim = 13

        layers: list[nn.Module] = []
        input_dim = feature_dim
        for _ in range(depth):
            layers.append(nn.Linear(input_dim, d_model))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = d_model
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""

        ee_xyz = state[:, 0:3]
        gripper = state[:, 3:4]
        red_xyz = state[:, 4:7]
        green_xyz = state[:, 7:10]
        blue_xyz = state[:, 10:13]
        goal_onehot = state[:, 13:16]
        goal_pos = state[:, 16:19]

        cubes = torch.stack([red_xyz, green_xyz, blue_xyz], dim=1)
        goal_idx = goal_onehot.argmax(dim=1)
        batch_idx = torch.arange(state.shape[0], device=state.device)
        target_xyz = cubes[batch_idx, goal_idx]
        target_rel_to_ee = target_xyz - ee_xyz
        bin_rel_to_target = goal_pos - target_xyz

        features = torch.cat(
            [
                ee_xyz,
                gripper,
                target_rel_to_ee,
                bin_rel_to_target,
                goal_onehot,
            ],
            dim=1,
        )
        return self.mlp(features).view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim,
    action_dim,
    chunk_size=16,
    d_model=256,
    depth=3,
    dropout=0.0,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            # DONE: Build with your chosen specifications
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            # DONE: Build with your chosen specifications
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")

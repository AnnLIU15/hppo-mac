"""H-PPO training utilities built on top of TorchRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import (
    InteractionType,
    CompositeDistribution,
    TensorDictModule,
    TensorDictSequential,
    set_composite_lp_aggregate,
)
from torch.distributions import Beta, OneHotCategorical
from torch.optim import Optimizer
from torchrl.collectors import SyncDataCollector
from torchrl.envs import TransformedEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.value import GAE
import numpy as np
from env.mac_simulator import MAC_PROTOCOL_COUNT, PREAMBLE_SUBSET_COUNT, HISTORY_DIM


_REQUEST_FEATURE_INDICES = {"cbra": 0, "pbra": 1}
_MAX_REQUEST_FEATURE_INDEX = max(_REQUEST_FEATURE_INDICES.values())
_PREAMBLE_FEATURE_START = 4 + MAC_PROTOCOL_COUNT + PREAMBLE_SUBSET_COUNT
_PREAMBLE_FEATURE_END = _PREAMBLE_FEATURE_START + 3
_ACB_FEATURE_INDEX = _PREAMBLE_FEATURE_END
_SUCCESS_FEATURE_INDEX = _ACB_FEATURE_INDEX + 1 + HISTORY_DIM
_COLLISION_FEATURE_INDEX = _SUCCESS_FEATURE_INDEX + 1
_PREAMBLE_KEYS = ("cbra", "pbra", "cfra")


class _PolicyParamExtractor(nn.Module):
    """Maps latent features to distribution parameters for each action head."""

    def __init__(self, feature_dim: int, combo_bins: int, *, mask_penalty: float = -1e9) -> None:
        super().__init__()
        self.combo_head = nn.Linear(feature_dim, combo_bins)
        self.mask_penalty = float(mask_penalty)
        # Shared head emits raw alpha/beta scores for the Beta distribution.
        self.acb_head = nn.Linear(feature_dim, 2)

    def forward(
        self,
        features: torch.Tensor,
        delta_combo_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        logits_combo = self.combo_head(features)
        if delta_combo_mask is not None:
            mask = delta_combo_mask.to(dtype=logits_combo.dtype, device=logits_combo.device)
            while mask.ndim < logits_combo.ndim:
                mask = mask.unsqueeze(0)
            mask = mask.expand_as(logits_combo)
            penalty = torch.full_like(logits_combo, self.mask_penalty)
            logits_combo = torch.where(mask > 0.5, logits_combo, penalty)
        acb_raw = self.acb_head(features)
        alpha = F.softplus(acb_raw[..., :1]) + 1.0
        beta = F.softplus(acb_raw[..., 1:2]) + 1.0
        return logits_combo, alpha, beta


class _FeatureEncoder(nn.Module):
    """Simple MLP encoder for flattened observations."""

    def __init__(self, in_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class _CriticNetwork(nn.Module):
    """Aggregates observation features into a scalar value estimate."""

    def __init__(self, in_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.encoder = _FeatureEncoder(in_dim, feature_dim)
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.encoder(observation)
        return self.value_head(features)


@dataclass
class HPPOConfig:
    """Hyper-parameters driving the H-PPO training loop."""

    frames_per_batch: int = 1600
    mini_batch_size: int = 64
    rollout_epochs: int = 10
    max_iterations: int = 200
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    device: Optional[torch.device] = None


def build_hppo_modules(
    env: TransformedEnv,
    *,
    feature_dim: int = 128,
) -> tuple[ProbabilisticActor, ValueOperator]:
    """Constructs probabilistic actor and critic operators aligned with the env specs."""

    observation_spec = env.observation_spec["observation"]
    action_spec = env.action_spec
    obs_dim = observation_spec.shape[-1]

    combo_spec = action_spec.get("delta_combo")
    if combo_spec is None:
        raise KeyError("Combined RA delta head `delta_combo` missing from action spec.")

    try:
        combo_bins = int(combo_spec.space.n)
    except AttributeError as err:
        raise ValueError("Discrete heads must expose a categorical cardinality via `space.n`.") from err

    actor_encoder = TensorDictModule(
        _FeatureEncoder(obs_dim, feature_dim),
        in_keys=["observation"],
        out_keys=["state_feature"],
    )
    policy_params = TensorDictModule(
        _PolicyParamExtractor(feature_dim, combo_bins),
        in_keys=["state_feature", ("info", "delta_combo_mask")],
        out_keys=[
            ("params", "delta_combo", "logits"),
            ("params", "q_ACB", "concentration1"),
            ("params", "q_ACB", "concentration0"),
        ],
    )

    policy_td = TensorDictSequential(actor_encoder, policy_params)

    # Ensure log-probs from composite distribution aggregate across heads.
    set_composite_lp_aggregate(True)

    actor = ProbabilisticActor(
        module=policy_td,
        in_keys=["params"],
        spec=action_spec,
        distribution_class=CompositeDistribution,
        distribution_kwargs={
            "distribution_map": {
                "delta_combo": OneHotCategorical,
                "q_ACB": Beta,
            }
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True,
    )

    critic = ValueOperator(
        module=TensorDictModule(
            _CriticNetwork(obs_dim, feature_dim),
            in_keys=["observation"],
            out_keys=["state_value"],
        ),
        in_keys=["observation"],
    )

    return actor, critic


def _prepare_optimizers(
    actor: ProbabilisticActor,
    critic: ValueOperator,
    config: HPPOConfig,
) -> tuple[Optimizer, Optimizer]:
    actor_opt = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    return actor_opt, critic_opt


def train_hppo(
    env: TransformedEnv,
    actor: ProbabilisticActor,
    critic: ValueOperator,
    config: HPPOConfig,
    *,
    logger_fn: Optional[callable] = None,
) -> Dict[str, float]:
    """Runs a TorchRL PPO loop tailored for the Satellite MAC environment."""

    device = config.device or torch.device("cpu")
    actor.to(device)
    critic.to(device)

    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.frames_per_batch * config.max_iterations,
        split_trajs=True,
        device=device,
    )

    advantage = GAE(
        gamma=config.gamma,
        lmbda=config.gae_lambda,
        value_network=critic,
        average_gae=False,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=config.clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=config.entropy_coeff,
        normalize_advantage=True,
    )

    actor_opt, critic_opt = _prepare_optimizers(actor, critic, config)

    metrics: Dict[str, float] = {}

    for iteration, batch in enumerate(collector):
        if iteration >= config.max_iterations:
            break

        with torch.no_grad():
            batch = advantage(batch)

        train_batch = batch.view(-1)
        train_batch = train_batch.to(device)

        batch_size = train_batch.batch_size[0]
        for epoch in range(config.rollout_epochs):
            permutation = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, config.mini_batch_size):
                idx = permutation[start : start + config.mini_batch_size]
                mini_batch = train_batch[idx]
                losses_td = loss_module(mini_batch)
                total_loss = losses_td["loss_objective"]

                entropy_term = losses_td.get("loss_entropy")
                if isinstance(entropy_term, torch.Tensor):
                    total_loss = total_loss + entropy_term

                critic_term = losses_td.get("loss_critic")
                if isinstance(critic_term, torch.Tensor):
                    total_loss = total_loss + critic_term

                actor_opt.zero_grad()
                critic_opt.zero_grad()
                total_loss.backward()
                actor_opt.step()
                critic_opt.step()

                scalar_metrics = {}
                for key, value in losses_td.items():
                    if isinstance(value, torch.Tensor):
                        scalar_metrics[key] = float(value.detach().cpu())
                metrics = scalar_metrics
        collector.update_policy_weights_()

        success_rate: Optional[float] = None
        collision_rate: Optional[float] = None
        acb_mean: Optional[float] = None
        preamble_means: Optional[Dict[str, float]] = None
        request_means: Optional[Dict[str, float]] = None
        try:
            obs_tensor = batch.get(("next", "observation"))
        except KeyError:
            obs_tensor = None

        if isinstance(obs_tensor, torch.Tensor):
            feature_dim = obs_tensor.shape[-1]
            flat_obs = obs_tensor.reshape(-1, feature_dim)
            if feature_dim > _COLLISION_FEATURE_INDEX:
                success_vals = flat_obs[:, _SUCCESS_FEATURE_INDEX]
                collision_vals = flat_obs[:, _COLLISION_FEATURE_INDEX]
                success_mean = float(success_vals.mean().detach().cpu().item())
                collision_mean = float(collision_vals.mean().detach().cpu().item())
                denom = success_mean + collision_mean
                if denom > 0.0:
                    success_rate = success_mean / denom
                    collision_rate = collision_mean / denom
                else:
                    success_rate = 0.0
                    collision_rate = 0.0
                metrics = {
                    **metrics,
                    "success_total_mean": success_mean,
                    "collision_total_mean": collision_mean,
                    "success_rate": success_rate,
                    "collision_rate": collision_rate,
                }
            if feature_dim > _ACB_FEATURE_INDEX:
                acb_vals = flat_obs[:, _ACB_FEATURE_INDEX]
                acb_mean = float(acb_vals.mean().detach().cpu().item())
                metrics = {**metrics, "acb_factor_mean": acb_mean}
            if feature_dim >= _PREAMBLE_FEATURE_END:
                preamble_slice = flat_obs[:, _PREAMBLE_FEATURE_START:_PREAMBLE_FEATURE_END]
                slice_mean = preamble_slice.mean(dim=0)
                preamble_means = {
                    f"preamble_{key}": np.round(value.detach().cpu().item() * 64)
                    for key, value in zip(_PREAMBLE_KEYS, slice_mean)
                }
                metrics = {**metrics, **preamble_means}
            if feature_dim > _MAX_REQUEST_FEATURE_INDEX:
                request_means = {
                    f"requests_{key}": float(flat_obs[:, idx].mean().detach().cpu().item())
                    for key, idx in _REQUEST_FEATURE_INDICES.items()
                }
                metrics = {**metrics, **request_means}

        if logger_fn:
            log_payload = {
                "iteration": iteration,
                "reward": float(batch["next", "reward"].mean().cpu().item()),
                **metrics,
            }
            if success_rate is not None and "success_rate" not in log_payload:
                log_payload["success_rate"] = success_rate
            if collision_rate is not None and "collision_rate" not in log_payload:
                log_payload["collision_rate"] = collision_rate
            if acb_mean is not None and "acb_factor_mean" not in log_payload:
                log_payload["acb_factor_mean"] = acb_mean
            if preamble_means is not None:
                for key, value in preamble_means.items():
                    if key not in log_payload:
                        log_payload[key] = value
            if request_means is not None:
                for key, value in request_means.items():
                    if key not in log_payload:
                        log_payload[key] = value

            logger_fn(**log_payload)

    collector.shutdown()
    return metrics

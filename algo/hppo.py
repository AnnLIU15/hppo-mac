"""H-PPO training utilities built on top of TorchRL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import (
    InteractionType,
    CompositeDistribution,
    TensorDictModule,
    TensorDictSequential,
    set_composite_lp_aggregate,
)
from torch.distributions import Distribution, Normal, OneHotCategorical, constraints
from torch.optim import Optimizer
from torchrl.collectors import SyncDataCollector
from torchrl.envs import TransformedEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives.ppo import ClipPPOLoss
from env.mac_simulator import (
    MAC_PROTOCOL_COUNT,
    PREAMBLE_SUBSET_COUNT,
)


_REQUEST_FEATURE_INDICES = {"cbra": 0, "pbra": 1}
_MAX_REQUEST_FEATURE_INDEX = max(_REQUEST_FEATURE_INDICES.values())
_PREAMBLE_FEATURE_START = 4 + MAC_PROTOCOL_COUNT + PREAMBLE_SUBSET_COUNT
_PREAMBLE_FEATURE_END = _PREAMBLE_FEATURE_START + 2
_ACB_FEATURE_INDEX = _PREAMBLE_FEATURE_END + 1


def _standardize_advantage(tensor: torch.Tensor, _: Sequence[int]) -> torch.Tensor:
    mean = tensor.mean()
    std = tensor.std(unbiased=False)
    std = torch.clamp(std, min=1e-8)
    return (tensor - mean) / std


def _sum_td_entries(td: TensorDictBase) -> torch.Tensor:
    total: Optional[torch.Tensor] = None
    for value in td.values():
        total = value if total is None else total + value
    if total is None:
        raise ValueError("TensorDict is empty; cannot aggregate entries.")
    return total


def _apply_reduction(value: torch.Tensor, reduction: Optional[str]) -> torch.Tensor:
    if reduction is None or reduction == "none":
        return value
    if reduction == "mean":
        return value.mean()
    if reduction == "sum":
        return value.sum()
    raise ValueError(f"Unsupported reduction mode: {reduction}")
def _compute_gae_targets(
    batch: TensorDictBase,
    critic: ValueOperator,
    *,
    gamma: float,
    gae_lambda: float,
) -> TensorDictBase:
    """Populate the batch TensorDict with GAE advantages and value targets."""

    with torch.no_grad():
        critic(batch)
        critic(batch.get("next"))

    rewards = batch.get(("next", "reward"))
    if rewards is None:
        raise KeyError("Missing reward tensor needed for advantage computation.")
    rewards = rewards.squeeze(-1)

    values = batch.get("state_value")
    if values is None:
        raise KeyError("Critic evaluation did not populate 'state_value'.")
    values = values.squeeze(-1)

    next_values = batch.get(("next", "state_value"))
    if next_values is None:
        raise KeyError("Critic evaluation did not populate 'next/state_value'.")
    next_values = next_values.squeeze(-1)
    next_values = torch.nan_to_num(next_values, nan=0.0, posinf=0.0, neginf=0.0)

    done = batch.get(("next", "done"))
    if done is None:
        terminated = batch.get(("next", "terminated"))
        truncated = batch.get(("next", "truncated"))
        if terminated is None or truncated is None:
            raise KeyError("Missing termination flags for GAE computation.")
        done = (terminated | truncated)
    done = done.to(dtype=values.dtype).squeeze(-1)

    original_shape = values.shape
    time_dim = values.shape[-1]

    values_flat = values.reshape(-1, time_dim)
    next_values_flat = next_values.reshape(-1, time_dim)
    rewards_flat = rewards.reshape(-1, time_dim)
    done_flat = done.reshape(-1, time_dim)

    advantages_flat = torch.zeros_like(values_flat)
    gae_accumulator = torch.zeros(values_flat.shape[0], dtype=values.dtype, device=values.device)

    for step in range(time_dim - 1, -1, -1):
        continuation = 1.0 - done_flat[:, step]
        bootstrap = next_values_flat[:, step] * continuation
        delta = rewards_flat[:, step] + bootstrap - values_flat[:, step]
        gae_accumulator = delta + gamma * gae_lambda * continuation * gae_accumulator
        advantages_flat[:, step] = gae_accumulator

    value_targets_flat = advantages_flat + values_flat

    advantages = advantages_flat.reshape(original_shape).unsqueeze(-1)
    value_targets = value_targets_flat.reshape(original_shape).unsqueeze(-1)

    batch.set("advantage", advantages)
    batch.set("value_target", value_targets)
    batch.set_("state_value", values.unsqueeze(-1))

    return batch


class TanhNormal01(Distribution):
    arg_constraints: Dict[str, constraints.Constraint] = {}
    support = constraints.interval(0.0, 1.0)
    has_rsample = True

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, *, min_log_std: float = -20.0, max_log_std: float = 2.0) -> None:
        self._mean = mean
        log_std = torch.clamp(log_std, min=min_log_std, max=max_log_std)
        self.log_std = log_std
        std = torch.exp(log_std)
        self._normal = Normal(mean, std)
        batch_shape = mean.shape
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size(), validate_args=False)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps
        x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _transform(self, sample: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(sample) + 1.0) * 0.5

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        base = self._normal.rsample(sample_shape)
        return self._transform(base)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        base = self._normal.sample(sample_shape)
        return self._transform(base)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(value.dtype).eps
        value = torch.clamp(value, eps, 1.0 - eps)
        tanh_value = value * 2.0 - 1.0
        pre_tanh = self._atanh(tanh_value)
        log_prob = self._normal.log_prob(pre_tanh)
        jacobian = torch.clamp(1.0 - tanh_value.pow(2), min=eps)
        log_det = torch.log(jacobian) - math.log(2.0)
        return log_prob - log_det

    @property
    def mean(self) -> torch.Tensor:
        return self._transform(self._mean)


_PREAMBLE_KEYS = ("cbra", "pbra", "cfra")


def _kaiming_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _PolicyParamExtractor(nn.Module):
    """Maps latent features to distribution parameters for each action head."""

    def __init__(self, feature_dim: int, delta_bins: int) -> None:
        super().__init__()
        self.cbra_head = nn.Linear(feature_dim, delta_bins)
        self.pbra_head = nn.Linear(feature_dim, delta_bins)
        self.acb_head = nn.Linear(feature_dim, 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        logits_cbra = self.cbra_head(features)
        logits_pbra = self.pbra_head(features)
        acb_params = self.acb_head(features)
        acb_mean, acb_log_std = acb_params.chunk(2, dim=-1)
        return logits_cbra, logits_pbra, acb_mean, acb_log_std


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


class SplitHeadClipPPOLoss(ClipPPOLoss):
    """PPO loss that treats each composite action head independently before aggregation."""

    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    def __init__(
        self,
        *args,
        head_keys: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._head_keys: Optional[tuple[str, ...]] = tuple(head_keys) if head_keys is not None else None

    @staticmethod
    def _normalized_head_log_probs(log_prob_td: TensorDictBase) -> TensorDict:
        """Return a TensorDict whose keys map cleanly to action head names."""

        entries = {}
        for key in log_prob_td.keys():
            value = log_prob_td.get(key)
            if isinstance(key, str) and key.endswith("_log_prob"):
                key = key[: -len("_log_prob")]
            entries[key] = value
        return TensorDict(
            entries,
            batch_size=log_prob_td.batch_size,
            device=log_prob_td.device,
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        tensordict = tensordict.clone(False)

        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            if self.critic_network is None:
                raise RuntimeError("Critic network is required to compute the advantage internally.")
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if advantage is None:
            raise KeyError(f"Missing advantage tensor at key {self.tensor_keys.advantage}.")

        if self.normalize_advantage and advantage.numel() > 1:
            advantage = _standardize_advantage(advantage, self.normalize_advantage_exclude_dims)

        previous_log_prob = tensordict.get("sample_log_prob")
        if previous_log_prob is None:
            log_prob_entries: Dict[str, torch.Tensor] = {}
            for key in tensordict.keys():
                if isinstance(key, str) and key.endswith("_log_prob"):
                    head_name = key[: -len("_log_prob")]
                    value = tensordict.get(key)
                    if value is not None:
                        log_prob_entries[head_name] = value
            if log_prob_entries:
                previous_log_prob = TensorDict(log_prob_entries, batch_size=tensordict.batch_size)
        if previous_log_prob is None:
            raise KeyError(f"Missing stored log-probs at key {self.tensor_keys.sample_log_prob}.")

        with set_composite_lp_aggregate(False):
            current_log_prob, dist, _ = self._get_cur_log_prob(tensordict)

        if isinstance(previous_log_prob, TensorDictBase):
            previous_log_prob = self._normalized_head_log_probs(previous_log_prob)
        if isinstance(current_log_prob, TensorDictBase):
            current_log_prob = self._normalized_head_log_probs(current_log_prob)

        if not isinstance(previous_log_prob, TensorDictBase) or not isinstance(current_log_prob, TensorDictBase):
            # Fallback to default PPO behaviour if composite heads are unavailable.
            return super().forward(tensordict)

        head_keys = self._head_keys or tuple(previous_log_prob.keys())
        clip_bounds = self._clip_bounds

        losses, log_weight_stack, kl_terms = [], [], []
        clip_fractions = []

        for key in head_keys:
            new_lp = current_log_prob.get(key)
            old_lp = previous_log_prob.get(key)
            if new_lp is None or old_lp is None:
                raise KeyError(f"Log-prob tensor for head '{key}' is missing.")

            if not torch.isfinite(new_lp).all() or not torch.isfinite(old_lp).all():
                stats = {
                    "new_nan": torch.isnan(new_lp).sum().item(),
                    "new_pos_inf": torch.isposinf(new_lp).sum().item(),
                    "new_neg_inf": torch.isneginf(new_lp).sum().item(),
                    "old_nan": torch.isnan(old_lp).sum().item(),
                    "old_pos_inf": torch.isposinf(old_lp).sum().item(),
                    "old_neg_inf": torch.isneginf(old_lp).sum().item(),
                }
                logits = tensordict.get(("params", key, "logits"), None)
                logits_stats = None
                if isinstance(logits, torch.Tensor):
                    logits_stats = {
                        "logits_nan": torch.isnan(logits).sum().item(),
                        "logits_pos_inf": torch.isposinf(logits).sum().item(),
                        "logits_neg_inf": torch.isneginf(logits).sum().item(),
                        "logits_min": float(logits.nan_to_num().min().detach().cpu().item()),
                        "logits_max": float(logits.nan_to_num().max().detach().cpu().item()),
                    }
                observation_stats = None
                obs = tensordict.get("observation", None)
                if isinstance(obs, torch.Tensor):
                    observation_stats = {
                        "obs_nan": torch.isnan(obs).sum().item(),
                        "obs_pos_inf": torch.isposinf(obs).sum().item(),
                        "obs_neg_inf": torch.isneginf(obs).sum().item(),
                        "obs_min": float(obs.nan_to_num().min().detach().cpu().item()),
                        "obs_max": float(obs.nan_to_num().max().detach().cpu().item()),
                    }
                raise RuntimeError(
                    f"Non-finite log-prob detected for head '{key}' with stats {stats}, logits_stats {logits_stats}, observation_stats {observation_stats}."
                )

            # Collapse potential event dimensions so each head operates on scalar log-probs.
            new_lp = new_lp.reshape(new_lp.shape[0], -1).sum(dim=-1)
            old_lp = old_lp.reshape(old_lp.shape[0], -1).sum(dim=-1)

            log_weight_scalar = new_lp - old_lp
            if torch.isfinite(log_weight_scalar).all():
                if (log_weight_scalar > 10000).any() or (log_weight_scalar < -10000).any():
                    raise RuntimeError(
                        f"Log-weight overflow for head '{key}'",
                        {
                            "max_log_weight": float(log_weight_scalar.max().detach().cpu().item()),
                            "min_log_weight": float(log_weight_scalar.min().detach().cpu().item()),
                            "max_new_lp": float(new_lp.max().detach().cpu().item()),
                            "min_new_lp": float(new_lp.min().detach().cpu().item()),
                            "max_old_lp": float(old_lp.max().detach().cpu().item()),
                            "min_old_lp": float(old_lp.min().detach().cpu().item()),
                        },
                    )

            log_weight = log_weight_scalar.unsqueeze(-1)
            log_weight_stack.append(log_weight)

            ratio = log_weight.exp()
            ratio_clipped = log_weight.clamp(*clip_bounds).exp()
            gain_candidates = torch.stack((ratio * advantage, ratio_clipped * advantage), dim=-1)
            losses.append(-gain_candidates.min(dim=-1).values)

            clip_flag = (log_weight.clamp(*clip_bounds) != log_weight).to(log_weight.dtype)
            clip_fractions.append(clip_flag.mean())

            kl_terms.append((old_lp - new_lp).unsqueeze(-1))

        loss_objective = torch.stack(losses, dim=-1).sum(dim=-1)
        clip_fraction_value = float(torch.stack(clip_fractions).mean().detach().cpu())
        clip_fraction = loss_objective.new_full(
            loss_objective.shape,
            clip_fraction_value,
        )
        kl_total = torch.stack(kl_terms, dim=-1).sum(dim=-1)

        td_out = TensorDict({"loss_objective": loss_objective})
        td_out.set("clip_fraction", clip_fraction)
        td_out.set("kl_approx", kl_total.detach())

        if self.entropy_bonus:
            entropy = self._get_entropy(dist, adv_shape=advantage.shape[:-1])
            if isinstance(entropy, TensorDictBase):
                td_out.set("composite_entropy", entropy.detach())
                entropy_tensor = _sum_td_entries(entropy).detach()
            else:
                entropy_tensor = entropy.detach()
            entropy_tensor = entropy_tensor.reshape(entropy_tensor.shape[0], -1).mean(dim=-1, keepdim=True)
            td_out.set("entropy", entropy_tensor)
            td_out.set("loss_entropy", self._weighted_loss_entropy(entropy))

        if self._has_critic:
            loss_critic, value_clip_fraction, explained_variance = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                if value_clip_fraction.shape[: len(tensordict.batch_size)] != tensordict.batch_size:
                    value_clip_fraction = float(value_clip_fraction.detach().mean().cpu())
                    value_clip_fraction = loss_objective.new_full(
                        loss_objective.shape,
                        value_clip_fraction,
                    )
                td_out.set("value_clip_fraction", value_clip_fraction)
            if explained_variance is not None:
                if explained_variance.shape[: len(tensordict.batch_size)] != tensordict.batch_size:
                    explained_variance = float(explained_variance.detach().mean().cpu())
                    explained_variance = loss_objective.new_full(
                        loss_objective.shape,
                        explained_variance,
                    )
                td_out.set("explained_variance", explained_variance)

        if log_weight_stack:
            with torch.no_grad():
                summed_log_weight = torch.stack([lw.squeeze(-1) for lw in log_weight_stack], dim=-1).sum(dim=-1)
                lw_flat = summed_log_weight.reshape(-1)
                ess = (2 * lw_flat.logsumexp(0) - (2 * lw_flat).logsumexp(0)).exp()
                batch = summed_log_weight.shape[0]
                ess_value = float((ess / batch).detach().cpu())
                ess_tensor = loss_objective.new_full(
                    loss_objective.shape,
                    ess_value,
                )
                td_out.set("ESS", ess_tensor)
        else:
            td_out.set("ESS", torch.zeros_like(loss_objective))

        td_out = td_out.named_apply(
            lambda name, value: _apply_reduction(value, self.reduction)
            if name.startswith("loss_")
            else value,
        )

        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
            "critic_network_params",
            "target_actor_network_params",
            "target_critic_network_params",
        )
        return td_out


def build_hppo_modules(
    env: TransformedEnv,
    *,
    feature_dim: int = 128,
) -> tuple[ProbabilisticActor, ValueOperator]:
    """Constructs probabilistic actor and critic operators aligned with the env specs."""

    observation_spec = env.observation_spec["observation"]
    action_spec = env.action_spec
    obs_dim = observation_spec.shape[-1]
    delta_spec = action_spec.get("delta_cbra")
    if delta_spec is None:
        raise KeyError("Discrete head `delta_cbra` missing from action spec.")

    try:
        delta_bins = int(delta_spec.space.n)
    except AttributeError as err:
        raise ValueError("Discrete heads must expose a categorical cardinality via `space.n`.") from err

    feature_encoder = _FeatureEncoder(obs_dim, feature_dim)
    feature_encoder.apply(_kaiming_init)
    actor_encoder = TensorDictModule(
        feature_encoder,
        in_keys=["observation"],
        out_keys=["state_feature"],
    )
    param_extractor = _PolicyParamExtractor(feature_dim, delta_bins)
    param_extractor.apply(_kaiming_init)
    policy_params = TensorDictModule(
        param_extractor,
        in_keys=["state_feature"],
        out_keys=[
            ("params", "delta_cbra", "logits"),
            ("params", "delta_pbra", "logits"),
            ("params", "q_ACB", "mean"),
            ("params", "q_ACB", "log_std"),
        ],
    )

    policy_td = TensorDictSequential(actor_encoder, policy_params)

    # Keep per-head log-prob components to facilitate split PPO losses.
    set_composite_lp_aggregate(False)

    actor = ProbabilisticActor(
        module=policy_td,
        in_keys=["params"],
        spec=action_spec,
        distribution_class=CompositeDistribution,
        distribution_kwargs={
            "distribution_map": {
                "delta_cbra": OneHotCategorical,
                "delta_pbra": OneHotCategorical,
                "q_ACB": TanhNormal01,
            }
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True,
    )

    critic = ValueOperator(
        module=TensorDictModule(
            _CriticNetwork(obs_dim, feature_dim).apply(_kaiming_init),
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
) -> tuple[Dict[str, float], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Runs a TorchRL PPO loop tailored for the Satellite MAC environment.

    Returns:
        tuple containing:
        - metrics: Dictionary of final training metrics
        - best_model_state: State dict of the best performing model (detached and copied)
        - last_model_state: State dict of the final model (detached and copied)
    """

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

    loss_module = SplitHeadClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=config.clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=config.entropy_coeff,
        normalize_advantage=True,
    head_keys=("delta_cbra", "delta_pbra", "q_ACB"),
    )

    actor_opt, critic_opt = _prepare_optimizers(actor, critic, config)

    metrics: Dict[str, float] = {}

    # 追踪最佳模型
    best_reward: float = float('-inf')
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    all_episode = len(collector)
    for iteration, batch in enumerate(collector):
        if iteration >= config.max_iterations:
            break

        with torch.no_grad():
            batch = _compute_gae_targets(
                batch,
                critic,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

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
                    # critic_term = torch.clamp(critic_term, max=100.0)
                    total_loss = total_loss + critic_term

                if not torch.isfinite(total_loss).all():
                    objective = losses_td.get("loss_objective")
                    entropy_loss = losses_td.get("loss_entropy")
                    critic_loss = losses_td.get("loss_critic")
                    raise RuntimeError(
                        "Encountered non-finite total loss",
                        {
                            "loss_objective_nan": 0 if not isinstance(objective, torch.Tensor) else int(torch.isnan(objective).sum().item()),
                            "loss_objective_inf": 0 if not isinstance(objective, torch.Tensor) else int(torch.isinf(objective).sum().item()),
                            "loss_entropy_nan": 0 if not isinstance(entropy_loss, torch.Tensor) else int(torch.isnan(entropy_loss).sum().item()),
                            "loss_entropy_inf": 0 if not isinstance(entropy_loss, torch.Tensor) else int(torch.isinf(entropy_loss).sum().item()),
                            "loss_critic_nan": 0 if not isinstance(critic_loss, torch.Tensor) else int(torch.isnan(critic_loss).sum().item()),
                            "loss_critic_inf": 0 if not isinstance(critic_loss, torch.Tensor) else int(torch.isinf(critic_loss).sum().item()),
                        },
                    )

                actor_opt.zero_grad()
                critic_opt.zero_grad()
                total_loss.backward()
                actor_opt.step()
                critic_opt.step()

                scalar_metrics = {}
                for key, value in losses_td.items():
                    if isinstance(value, torch.Tensor):
                        reduced = value
                        if reduced.numel() > 1:
                            reduced = reduced.mean()
                        scalar_metrics[key] = float(reduced.detach().cpu())
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
            if feature_dim > _ACB_FEATURE_INDEX:
                acb_vals = flat_obs[:, _ACB_FEATURE_INDEX]
                acb_mean = float(acb_vals.mean().detach().cpu().item())
                metrics = {**metrics, "acb_mean": acb_mean}
            if feature_dim > _MAX_REQUEST_FEATURE_INDEX:
                request_means = {
                    f"r_{key}": float(flat_obs[:, idx].mean().detach().cpu().item())
                    for key, idx in _REQUEST_FEATURE_INDICES.items()
                }
                metrics = {**metrics, **request_means}
            if feature_dim > _PREAMBLE_FEATURE_END:
                preamble_means = {
                    f"p_{key}": float(
                        flat_obs[:, _PREAMBLE_FEATURE_START + i][:-1].mean().detach().cpu().item()
                    )
                    for i, key in enumerate(_PREAMBLE_KEYS)
                }
                metrics = {**metrics, **preamble_means}

        # 获取当前迭代的平均奖励
        current_reward = float(batch["next", "reward"].mean().cpu().item())

        # 更新最佳模型（如果当前奖励更好）
        if (current_reward > best_reward) and (iteration > all_episode//2):
            best_reward = current_reward
            # 深拷贝并detach actor和critic的state_dict
            best_model_state = {
                'actor': {k: v.detach().clone().cpu() for k, v in actor.state_dict().items()},
                'critic': {k: v.detach().clone().cpu() for k, v in critic.state_dict().items()},
                'iteration': iteration,
                'reward': current_reward,
            }

        if logger_fn:
            log_payload = {
                "iteration": iteration,
                "reward": current_reward,
                **metrics,
            }
            if success_rate is not None and "success_rate" not in log_payload:
                log_payload["succe_rate"] = success_rate
            if collision_rate is not None and "collision_rate" not in log_payload:
                log_payload["coll_rate"] = collision_rate
            if acb_mean is not None and "acb_factor_mean" not in log_payload:
                log_payload["acb_mean"] = acb_mean
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

    # 保存最后一轮模型的state_dict（深拷贝并detach）
    last_model_state = {
        'actor': {k: v.detach().clone().cpu() for k, v in actor.state_dict().items()},
        'critic': {k: v.detach().clone().cpu() for k, v in critic.state_dict().items()},
        'iteration': iteration,
        'reward': current_reward if 'current_reward' in locals() else float('nan'),
    }

    # 如果没有找到更好的模型（比如只运行了0次迭代），使用最后的模型作为最佳模型
    if best_model_state is None:
        best_model_state = last_model_state

    return metrics, best_model_state, last_model_state

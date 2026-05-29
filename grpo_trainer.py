import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
import config
from rl_utils import build_port_state


class GRPOTrainer:
    def __init__(self, encoder, policy, env,
                 G=config.GRPO_G, beta=config.GRPO_BETA,
                 lr_policy=config.LR_POLICY, lr_encoder=config.LR_ENCODER,
                 ref_refresh_steps=config.GRPO_REF_REFRESH,
                 encoder_unfreeze_step=config.ENCODER_UNFREEZE_STEP,
                 device="cpu"):
        self.encoder = encoder
        self.policy = policy
        self.env = env
        self.G = G
        self.beta = beta
        self.ref_refresh_steps = ref_refresh_steps
        self.encoder_unfreeze_step = encoder_unfreeze_step
        self.device = device
        self.step_count = 0

        self.ref_policy = copy.deepcopy(policy)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        self.optimizer_policy = torch.optim.Adam(
            policy.parameters(), lr=lr_policy)
        self.optimizer_encoder = torch.optim.Adam(
            encoder.parameters(), lr=lr_encoder)

        for p in encoder.parameters():
            p.requires_grad = False
        self.encoder_frozen = True

    def _maybe_unfreeze_encoder(self):
        if self.encoder_frozen and self.step_count >= self.encoder_unfreeze_step:
            for name, p in self.encoder.named_parameters():
                if not name.startswith('denoiser.'):
                    p.requires_grad = True
            self.encoder_frozen = False

    def _maybe_refresh_ref(self):
        if self.step_count % self.ref_refresh_steps == 0 and self.step_count > 0:
            self.ref_policy.load_state_dict(self.policy.state_dict())

    # === PLACEHOLDER_GRPO_COLLECT ===

    def collect_and_update(self, dynamic_x, static_x, port_state, mask, head):
        self._maybe_unfreeze_encoder()
        self._maybe_refresh_ref()

        with torch.set_grad_enabled(not self.encoder_frozen):
            enc = self.encoder(dynamic_x, static_x)

        dist = self.policy(enc, port_state, mask, head)

        log_probs_all = []
        rewards = []
        actions_all = []

        n_bins = config.N_BINS
        bins = torch.tensor(config.BINS, device=self.device, dtype=torch.float32)

        for _ in range(self.G):
            action = dist.sample()
            actions_all.append(action)
            log_probs_all.append(dist.log_prob(action))

            weights = bins[action].detach().cpu().numpy()
            noise = np.random.uniform(0, 1e-6, size=weights.shape)
            top_k_idx = np.argsort(weights + noise)[-config.N_HOLD:]
            target_w = np.zeros(self.env.n_stocks)
            for idx in top_k_idx:
                if idx < len(weights):
                    target_w[idx] = weights[idx]
            w_sum = target_w.sum()
            if w_sum > 0:
                target_w = target_w / w_sum

            env_clone = self.env.clone()
            _, reward, _, _ = env_clone.step(target_w)
            rewards.append(reward)

        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        reward_std = rewards.std()
        if reward_std < 1e-4:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - rewards.mean()) / (reward_std + 1e-8)

        log_probs = torch.stack(log_probs_all)
        if log_probs.dim() > 1:
            log_probs = log_probs.mean(dim=-1)

        with torch.no_grad():
            ref_dist = self.ref_policy(enc.detach(), port_state, mask, head)

        kl = torch.distributions.kl_divergence(dist, ref_dist)
        if kl.dim() > 1:
            kl = kl.mean(dim=-1)
        kl = kl.mean()

        policy_loss = -(advantages.detach() * log_probs).mean()
        loss = policy_loss + self.beta * kl

        self.optimizer_policy.zero_grad()
        if not self.encoder_frozen:
            self.optimizer_encoder.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer_policy.step()
        if not self.encoder_frozen:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.encoder.parameters() if p.requires_grad],
                1.0)
            self.optimizer_encoder.step()

        self.step_count += 1

        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "kl": kl.item(),
        }

    def collect_trajectory_and_update(self, env, obs_cache, start_idx, device):
        """Run G full episodes from start_idx, compute trajectory-level GRPO."""
        self._maybe_unfreeze_encoder()
        self._maybe_refresh_ref()

        bins = torch.tensor(config.BINS, device=device, dtype=torch.float32)
        init_env = env.clone()

        trajectory_rewards = []
        trajectory_log_probs = []

        for g in range(self.G):
            env_g = init_env.clone()
            env_g.reset(start_date_idx=start_idx,
                        episode_len=config.EPISODE_LEN)
            total_reward = 0.0
            total_log_prob = torch.tensor(0.0, device=device)
            done = False

            for day in range(config.EPISODE_LEN):
                if done:
                    break
                date_idx = env_g.current_idx
                for phase in ["open", "close"]:
                    env_g.phase = phase
                    dyn_t, stat_t, mask_t = obs_cache.get_obs(
                        date_idx, env_g, device)
                    port_state = build_port_state(env_g, device)

                    with torch.no_grad():
                        enc = self.encoder(dyn_t, stat_t)
                    dist = self.policy(enc.detach(), port_state, mask_t, phase)
                    action = dist.sample()
                    lp = dist.log_prob(action)
                    if lp.dim() > 0:
                        lp = lp.mean()
                    total_log_prob = total_log_prob + lp

                    weights = bins[action].detach().cpu().numpy()
                    top_k_idx = np.argsort(weights)[-config.N_HOLD:]
                    target_w = np.zeros(env_g.n_stocks)
                    for idx in top_k_idx:
                        target_w[idx] = weights[idx]
                    w_sum = target_w.sum()
                    if w_sum > 0:
                        target_w = target_w / w_sum

                    _, reward, done, _ = env_g.step(target_w)
                    total_reward += reward
                    if done:
                        break

            trajectory_rewards.append(total_reward)
            trajectory_log_probs.append(total_log_prob)

        rewards_t = torch.tensor(trajectory_rewards, device=device,
                                 dtype=torch.float32)
        reward_std = rewards_t.std()
        if reward_std < 1e-6:
            advantages = torch.zeros_like(rewards_t)
        else:
            advantages = (rewards_t - rewards_t.mean()) / (reward_std + 1e-8)

        log_probs_t = torch.stack(trajectory_log_probs)

        with torch.no_grad():
            dyn_t, stat_t, mask_t = obs_cache.get_obs(
                start_idx, init_env, device)
            enc_ref = self.encoder(dyn_t, stat_t)
            port_state = build_port_state(init_env, device)
            cur_dist = self.policy(enc_ref, port_state, mask_t, "open")
            ref_dist = self.ref_policy(enc_ref, port_state, mask_t, "open")
        kl = torch.distributions.kl_divergence(cur_dist, ref_dist)
        if kl.dim() > 1:
            kl = kl.mean(dim=-1)
        kl = kl.mean()

        policy_loss = -(advantages.detach() * log_probs_t).mean()
        loss = policy_loss + self.beta * kl

        self.optimizer_policy.zero_grad()
        if not self.encoder_frozen:
            self.optimizer_encoder.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer_policy.step()
        if not self.encoder_frozen:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.encoder.parameters() if p.requires_grad],
                1.0)
            self.optimizer_encoder.step()

        self.step_count += 1
        return {
            "loss": loss.item(),
            "mean_reward": rewards_t.mean().item(),
            "best_reward": rewards_t.max().item(),
            "kl": kl.item(),
        }
import copy
import numpy as np
import torch
import config
from rl_utils import build_port_state


class GRPOTrainer:
    def __init__(self, encoder, policy, env,
                 G=config.GRPO_G, beta=config.GRPO_BETA,
                 lr_policy=config.LR_POLICY, lr_encoder=config.LR_ENCODER,
                 ref_refresh_steps=config.GRPO_REF_REFRESH,
                 device="cpu"):
        self.encoder = encoder
        self.policy = policy
        self.env = env
        self.G = G
        self.beta = beta
        self.ref_refresh_steps = ref_refresh_steps
        self.device = device
        self.step_count = 0
        self.use_amp = (device != "cpu" and
                        torch.cuda.is_available())
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.ref_policy = copy.deepcopy(policy)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        self.optimizer_policy = torch.optim.Adam(
            policy.parameters(), lr=lr_policy)
        self.optimizer_encoder = torch.optim.Adam(
            encoder.parameters(), lr=lr_encoder)

    def _maybe_refresh_ref(self):
        if self.step_count % self.ref_refresh_steps == 0 and self.step_count > 0:
            self.ref_policy.load_state_dict(self.policy.state_dict())

    def collect_trajectory_and_update(self, env, obs_cache, start_idx, device):
        """Run G full episodes from start_idx, compute trajectory-level GRPO.

        KL divergence is computed per-step across the trajectory and averaged.
        Noise is sampled once per (date_idx, phase) and shared across G trajectories.
        """
        self._maybe_refresh_ref()

        bins = torch.tensor(config.BINS, device=device, dtype=torch.float32)
        init_env = env.clone()

        trajectory_rewards = []
        trajectory_log_probs = []
        trajectory_kls = []
        noise_cache = {}

        for g in range(self.G):
            env_g = init_env.clone()
            env_g.reset(start_date_idx=start_idx,
                        episode_len=config.EPISODE_LEN)
            total_reward = 0.0
            total_log_prob = torch.tensor(0.0, device=device)
            total_kl = torch.tensor(0.0, device=device)
            n_steps = 0
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

                    cache_key = (date_idx, phase)
                    if cache_key not in noise_cache:
                        noise_cache[cache_key] = torch.randn_like(dyn_t)
                    shared_noise = noise_cache[cache_key]

                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        enc = self.encoder(dyn_t, stat_t,
                                           denoise_noise=shared_noise)
                        cur_dist = self.policy(enc, port_state, mask_t, phase)
                    action = cur_dist.sample()
                    lp = cur_dist.log_prob(action)
                    if lp.dim() > 0:
                        lp = lp.mean()
                    total_log_prob = total_log_prob + lp

                    with torch.no_grad():
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            ref_dist = self.ref_policy(
                                enc.detach(), port_state, mask_t, phase)
                    step_kl = torch.distributions.kl_divergence(
                        cur_dist, ref_dist)
                    if step_kl.dim() > 1:
                        step_kl = step_kl.mean(dim=-1)
                    total_kl = total_kl + step_kl.mean()
                    n_steps += 1

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
            trajectory_kls.append(total_kl / max(n_steps, 1))

        rewards_t = torch.tensor(trajectory_rewards, device=device,
                                 dtype=torch.float32)
        reward_std = rewards_t.std()
        if reward_std < 1e-6:
            advantages = torch.zeros_like(rewards_t)
        else:
            advantages = (rewards_t - rewards_t.mean()) / (reward_std + 1e-8)

        log_probs_t = torch.stack(trajectory_log_probs)
        kl_avg = torch.stack(trajectory_kls).mean()

        policy_loss = -(advantages.detach() * log_probs_t).mean()
        loss = policy_loss + self.beta * kl_avg

        self.optimizer_policy.zero_grad()
        self.optimizer_encoder.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer_policy)
            self.scaler.unscale_(self.optimizer_encoder)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.scaler.step(self.optimizer_policy)
            self.scaler.step(self.optimizer_encoder)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer_policy.step()
            self.optimizer_encoder.step()

        self.step_count += 1
        return {
            "loss": loss.item(),
            "mean_reward": rewards_t.mean().item(),
            "best_reward": rewards_t.max().item(),
            "kl": kl_avg.item(),
        }

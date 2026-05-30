import copy
import numpy as np
import torch
import torch.nn.functional as F
import config
from rl_utils import build_port_state


class GRPOTrainer:
    def __init__(self, encoder, policy, env, return_predictor=None,
                 G=config.GRPO_G, beta=config.GRPO_BETA,
                 lr_policy=config.LR_POLICY, lr_encoder=config.LR_ENCODER,
                 ref_refresh_steps=config.GRPO_REF_REFRESH,
                 device="cpu"):
        self.encoder = encoder
        self.policy = policy
        self.env = env
        self.return_predictor = return_predictor
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
        if return_predictor is not None:
            self.optimizer_aux = torch.optim.Adam(
                return_predictor.parameters(), lr=lr_policy)

    def _maybe_refresh_ref(self):
        if self.step_count % self.ref_refresh_steps == 0 and self.step_count > 0:
            self.ref_policy.load_state_dict(self.policy.state_dict())

    def _print_diag(self, rewards_t, advantages, action_hashes,
                    policy_loss, kl_avg, aux_loss_val,
                    policy_gnorm, enc_gnorm):
        r = rewards_t.detach().float()
        a = advantages.detach().float()
        n_unique = len(set(action_hashes)) if action_hashes else 0
        pl = policy_loss.item()
        kl_term = self.beta * kl_avg.item()
        aux_term = config.LAMBDA_AUX * aux_loss_val
        print(f"  [DIAG step {self.step_count}]"
              + ("  (SMOKE)" if config.DIAG_SMOKE_TEST else ""))
        print(f"    D1 reward   : mean={r.mean():.6f} std={r.std():.6f} "
              f"min={r.min():.6f} max={r.max():.6f}")
        print(f"    D1 advantage: mean={a.mean():.6f} std={a.std():.6f} "
              f"min={a.min():.6f} max={a.max():.6f}")
        print(f"    D2 traj div : {n_unique}/{self.G} unique action seqs")
        print(f"    D3 grad norm: encoder={enc_gnorm:.6f} "
              f"policy={policy_gnorm:.6f}")
        print(f"    D4 loss part: policy={pl:.6f} "
              f"kl_term={kl_term:.6f} aux_term={aux_term:.6f}")

    def collect_trajectory_and_update(self, env, obs_cache, start_idx, device):
        """Run G full episodes from start_idx, compute trajectory-level GRPO.

        KL divergence is computed per-step across the trajectory and averaged.
        Noise is sampled once per (date_idx, phase) and shared across G trajectories.
        """
        self._maybe_refresh_ref()

        diag_on = (config.DIAG_INTERVAL > 0 and
                   self.step_count % config.DIAG_INTERVAL == 0)

        bins = torch.tensor(config.BINS, device=device, dtype=torch.float32)
        init_env = env.clone()

        # DIAG5: smoke test 用确定性合成 alpha 替代真实奖励，隔离金融噪声
        if config.DIAG_SMOKE_TEST and not hasattr(self, "_synth_alpha"):
            n = init_env.n_stocks
            self._synth_alpha = np.sin(np.arange(n) * 0.7).astype(np.float32)

        trajectory_rewards = []
        trajectory_log_probs = []
        trajectory_kls = []
        noise_cache = {}
        aux_enc_list = []
        aux_ret_list = []
        traj_action_hashes = []  # DIAG2: 每条轨迹的动作序列哈希

        for g in range(self.G):
            env_g = init_env.clone()
            env_g.reset(start_date_idx=start_idx,
                        episode_len=config.EPISODE_LEN)
            total_reward = 0.0
            total_log_prob = torch.tensor(0.0, device=device)
            total_kl = torch.tensor(0.0, device=device)
            n_steps = 0
            done = False
            traj_actions = []  # DIAG2: 本条轨迹的动作序列

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
                    if diag_on:
                        traj_actions.append(action.detach())
                    lp = cur_dist.log_prob(action)
                    if lp.dim() > 0:
                        lp = lp.mean()
                    total_log_prob = total_log_prob + lp

                    if self.return_predictor is not None and g == 0:
                        date_idx_cur = env_g.current_idx
                        if date_idx_cur > 0:
                            prev_close = env_g.close_prices[date_idx_cur - 1]
                            cur_close = env_g.close_prices[date_idx_cur]
                            stock_ret = cur_close / prev_close - 1.0
                            valid = np.isfinite(stock_ret)
                            if valid.any():
                                valid_t = torch.tensor(valid, device=device)
                                ret_t = torch.tensor(
                                    np.nan_to_num(stock_ret, 0.0),
                                    device=device, dtype=torch.float32)
                                aux_enc_list.append(enc.detach()[valid_t])
                                aux_ret_list.append(ret_t[valid_t])

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
                    if config.DIAG_SMOKE_TEST:
                        # 确定性合成奖励：组合权重与固定 alpha 的内积。
                        # 与动作强相关、无金融噪声，用于隔离测试管线能否学习。
                        reward = float(np.dot(target_w, self._synth_alpha))
                    total_reward += reward
                    if done:
                        break

            trajectory_rewards.append(total_reward)
            trajectory_log_probs.append(total_log_prob)
            trajectory_kls.append(total_kl / max(n_steps, 1))

            if diag_on and traj_actions:
                flat = torch.cat([a.reshape(-1) for a in traj_actions])
                traj_action_hashes.append(hash(flat.cpu().numpy().tobytes()))

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

        aux_loss_val = 0.0
        if self.return_predictor is not None and aux_enc_list:
            aux_enc = torch.cat(aux_enc_list, dim=0)
            aux_target = torch.cat(aux_ret_list, dim=0)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                aux_pred = self.return_predictor(aux_enc)
            aux_loss = F.mse_loss(aux_pred, aux_target)
            aux_loss_val = aux_loss.item()
            loss = loss + config.LAMBDA_AUX * aux_loss

        self.optimizer_policy.zero_grad()
        self.optimizer_encoder.zero_grad()
        if self.return_predictor is not None:
            self.optimizer_aux.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer_policy)
            self.scaler.unscale_(self.optimizer_encoder)
            if self.return_predictor is not None:
                self.scaler.unscale_(self.optimizer_aux)
            policy_gnorm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 1.0)
            enc_gnorm = torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), 1.0)
            self.scaler.step(self.optimizer_policy)
            self.scaler.step(self.optimizer_encoder)
            if self.return_predictor is not None:
                self.scaler.step(self.optimizer_aux)
            self.scaler.update()
        else:
            loss.backward()
            policy_gnorm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 1.0)
            enc_gnorm = torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), 1.0)
            self.optimizer_policy.step()
            self.optimizer_encoder.step()
            if self.return_predictor is not None:
                self.optimizer_aux.step()

        if diag_on:
            self._print_diag(
                rewards_t, advantages, traj_action_hashes,
                policy_loss, kl_avg, aux_loss_val,
                float(policy_gnorm), float(enc_gnorm))

        self.step_count += 1
        return {
            "loss": loss.item(),
            "mean_reward": rewards_t.mean().item(),
            "best_reward": rewards_t.max().item(),
            "kl": kl_avg.item(),
        }

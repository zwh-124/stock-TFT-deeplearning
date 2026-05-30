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
        trajectory_step_rewards = []  # #3: 每条轨迹的逐步 reward 列表 [G][T]
        trajectory_step_logps = []    # #4: 每条轨迹的逐步 per-stock log_prob [G][T][N_HOLD]
        trajectory_step_stock_returns = []  # #4: 每条轨迹的逐步 per-stock 收益 [G][T][N_HOLD]
        trajectory_kls = []
        noise_cache = {}
        aux_enc_list = []
        aux_ret_list = []
        traj_action_hashes = []  # DIAG2: 每条轨迹的动作序列哈希
        enc_cache = {}   # #1: (date_idx, phase) -> enc，G 条轨迹复用同一编码器前向
        mask_cache = {}  # #1: (date_idx, phase) -> mask_t（与持仓无关，可复用）

        for g in range(self.G):
            env_g = init_env.clone()
            env_g.reset(start_date_idx=start_idx,
                        episode_len=config.EPISODE_LEN)
            total_reward = 0.0
            step_logps = []   # #4: 本条轨迹每一步的 per-stock log_prob [N_HOLD]
            step_rewards = []  # #3: 本条轨迹每一步的即时 reward（float）
            step_stock_returns = []  # #4: 本条轨迹每一步的 per-stock 收益 [N_HOLD]
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
                    cache_key = (date_idx, phase)

                    # #1: encoder 前向只依赖 (date_idx, phase)，与持仓无关，
                    # 故 G 条轨迹复用同一个 enc / mask（编码器是最贵的部分）。
                    if cache_key in enc_cache:
                        enc = enc_cache[cache_key]
                        mask_t = mask_cache[cache_key]
                    else:
                        dyn_t, stat_t, mask_t = obs_cache.get_obs(
                            date_idx, env_g, device)
                        if cache_key not in noise_cache:
                            noise_cache[cache_key] = torch.randn_like(dyn_t)
                        shared_noise = noise_cache[cache_key]
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            enc = self.encoder(dyn_t, stat_t,
                                               denoise_noise=shared_noise)
                        enc_cache[cache_key] = enc
                        mask_cache[cache_key] = mask_t

                    # port_state 依赖持仓，必须逐轨迹计算（policy 是廉价 MLP）。
                    port_state = build_port_state(env_g, device)
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        cur_dist = self.policy(enc, port_state, mask_t, phase)
                    action = cur_dist.sample()
                    if diag_on:
                        traj_actions.append(action.detach())
                    # #2: 先取每只股票的 log_prob，待持仓确定后再筛选累加（见下方）
                    lp_all = cur_dist.log_prob(action)

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

                    # #4: 保留每只持仓股的 log_prob（不再 .mean()），
                    # 用于 per-stock 截面优势的逐股梯度分配。
                    hold_idx_t = torch.as_tensor(
                        np.asarray(top_k_idx), device=device, dtype=torch.long)
                    lp_per_stock = lp_all[hold_idx_t]  # [N_HOLD]
                    step_logps.append(lp_per_stock)

                    _, reward, done, info = env_g.step(target_w)
                    # #4: 收集 per-stock 实现收益（用于截面优势）
                    per_stock_ret = info.get('per_stock_returns',
                                            np.zeros(env_g.n_stocks))
                    held_returns = per_stock_ret[top_k_idx]  # [N_HOLD]
                    if config.DIAG_SMOKE_TEST:
                        reward = float(np.dot(target_w, self._synth_alpha))
                        held_returns = self._synth_alpha[top_k_idx]
                    step_stock_returns.append(held_returns)
                    step_rewards.append(reward)
                    total_reward += reward
                    if done:
                        break

            trajectory_rewards.append(total_reward)
            trajectory_step_rewards.append(step_rewards)
            trajectory_step_logps.append(step_logps)
            trajectory_step_stock_returns.append(step_stock_returns)
            trajectory_kls.append(total_kl / max(n_steps, 1))

            if diag_on and traj_actions:
                flat = torch.cat([a.reshape(-1) for a in traj_actions])
                traj_action_hashes.append(hash(flat.cpu().numpy().tobytes()))

        rewards_t = torch.tensor(trajectory_rewards, device=device,
                                 dtype=torch.float32)
        kl_avg = torch.stack(trajectory_kls).mean()

        # #4: per-stock 截面优势 + return-to-go 时间优势叠加。
        # 解决原 #3 的核心缺陷：同一步所有持仓股共享标量优势，无法区分个股贡献。
        gamma = config.GRPO_GAMMA
        T = min(len(sr) for sr in trajectory_step_rewards)
        N_HOLD_actual = config.N_HOLD
        if T == 0:
            policy_loss = torch.tensor(0.0, device=device)
            advantages = torch.zeros(self.G, 1, N_HOLD_actual, device=device)
        else:
            # --- A_temporal: [G, T] return-to-go 跨 G 轨迹 z-score ---
            returns_mat = torch.zeros(self.G, T, device=device,
                                      dtype=torch.float32)
            for g in range(self.G):
                sr = trajectory_step_rewards[g]
                acc = 0.0
                for t in range(T - 1, -1, -1):
                    acc = sr[t] + gamma * acc
                    returns_mat[g, t] = acc

            mean_g = returns_mat.mean(dim=0, keepdim=True)
            std_g = returns_mat.std(dim=0, keepdim=True)
            A_temporal = (returns_mat - mean_g) / (std_g + 1e-8)
            A_temporal = torch.where(std_g < 1e-6,
                                     torch.zeros_like(A_temporal), A_temporal)
            # 扩展到 [G, T, N_HOLD]（每只持仓股共享时间维度优势）
            A_temporal = A_temporal.unsqueeze(-1).expand(-1, -1, N_HOLD_actual)

            # --- A_cross: [G, T, N_HOLD] 截面优势，跨持仓股 z-score ---
            stock_ret_mat = torch.zeros(self.G, T, N_HOLD_actual,
                                        device=device, dtype=torch.float32)
            for g in range(self.G):
                for t in range(T):
                    stock_ret_mat[g, t] = torch.tensor(
                        trajectory_step_stock_returns[g][t],
                        device=device, dtype=torch.float32)

            cross_mean = stock_ret_mat.mean(dim=-1, keepdim=True)
            cross_std = stock_ret_mat.std(dim=-1, keepdim=True) + 1e-8
            A_cross = (stock_ret_mat - cross_mean) / cross_std
            A_cross = torch.where(cross_std < 1e-6,
                                  torch.zeros_like(A_cross), A_cross)

            # --- 叠加 ---
            alpha = config.ALPHA_TEMPORAL
            advantages = alpha * A_temporal + (1 - alpha) * A_cross

            # logp_mat: [G, T, N_HOLD]（每只股票独立的 log_prob）
            logp_mat = torch.stack(
                [torch.stack(trajectory_step_logps[g][:T])
                 for g in range(self.G)])
            policy_loss = -(advantages.detach() * logp_mat).mean()

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

import copy
import numpy as np
import torch
import torch.nn.functional as F
import config


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
            for p in self.encoder.parameters():
                p.requires_grad = True
            self.encoder_frozen = False

    def _maybe_refresh_ref(self):
        if self.step_count % self.ref_refresh_steps == 0 and self.step_count > 0:
            self.ref_policy.load_state_dict(self.policy.state_dict())

    # === PLACEHOLDER_GRPO_COLLECT ===

    def collect_and_update(self, dynamic_x, static_x, port_state, mask, head):
        """One GRPO step: sample G actions, simulate, compute loss, update.

        Args:
            dynamic_x: [N_alive, T, V] tensor
            static_x: [N_alive, S] tensor
            port_state: [N_alive, 4] tensor (cash_frac, hold_frac, locked_frac, prev_w)
            mask: [N_alive] bool tensor (tradeable stocks)
            head: "open" or "close"

        Returns:
            dict with loss, mean_reward, kl
        """
        self._maybe_unfreeze_encoder()
        self._maybe_refresh_ref()

        with torch.set_grad_enabled(not self.encoder_frozen):
            enc = self.encoder(dynamic_x, static_x)

        dist = self.policy(enc, port_state, mask, head)

        log_probs_all = []
        rewards = []
        actions_all = []

        base_state = self.env._get_state()
        n_bins = config.N_BINS
        bins = torch.tensor(config.BINS, device=self.device, dtype=torch.float32)

        for _ in range(self.G):
            action = dist.sample()
            actions_all.append(action)
            log_probs_all.append(dist.log_prob(action))

            weights = bins[action].detach().cpu().numpy()
            top_k_idx = np.argsort(weights)[-config.N_HOLD:]
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
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        log_probs = torch.stack(log_probs_all)
        if log_probs.dim() > 1:
            log_probs = log_probs.sum(dim=-1)

        with torch.no_grad():
            ref_dist = self.ref_policy(enc.detach(), port_state, mask, head)

        kl = torch.distributions.kl_divergence(dist, ref_dist)
        if kl.dim() > 1:
            kl = kl.sum(dim=-1)
        kl = kl.mean()

        policy_loss = -(advantages.detach() * log_probs).mean()
        loss = policy_loss + self.beta * kl

        self.optimizer_policy.zero_grad()
        if not self.encoder_frozen:
            self.optimizer_encoder.zero_grad()

        loss.backward()

        self.optimizer_policy.step()
        if not self.encoder_frozen:
            self.optimizer_encoder.step()

        self.step_count += 1

        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "kl": kl.item(),
        }
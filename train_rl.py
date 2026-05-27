import os
import random
import numpy as np
import torch

import config
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from dataset import rolling_normalize_window
from model import TFTEncoder, PortfolioPolicy, DiffusionDenoiser
from env import AShareTradingEnv
from grpo_trainer import GRPOTrainer


def prepare_data():
    print("Loading and preparing data...")
    df = build_merged_dataset(config.TRAIN_START, config.TEST_END)
    df, avail_features = build_features(df)
    return df, avail_features


def build_env(df):
    env = AShareTradingEnv(df)
    return env


def load_pretrained_encoder(encoder, device):
    ckpt_path = os.path.join(config.CACHE_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"WARNING: {ckpt_path} not found, using random init")
        return
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder_state = {}
    sd = state.get('state_dict', state)
    for k, v in sd.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v
        elif not k.startswith("fc_out"):
            encoder_state[k] = v
    encoder.load_state_dict(encoder_state, strict=False)
    print("Loaded pretrained encoder weights.")

    if config.USE_DIFFUSION_DENOISER and state.get('denoiser_state'):
        denoiser = DiffusionDenoiser(
            feature_dim=len(DYNAMIC_FEATURES),
            seq_len=config.SEQ_LEN,
            hidden_dim=config.DIFFUSION_HIDDEN_DIM,
            time_dim=config.DIFFUSION_TIME_DIM,
            n_timesteps=config.DIFFUSION_T,
            beta_start=config.DIFFUSION_BETA_START,
            beta_end=config.DIFFUSION_BETA_END,
        ).to(device)
        denoiser.load_state_dict(state['denoiser_state'])
        denoiser.eval()
        for p in denoiser.parameters():
            p.requires_grad = False
        encoder.denoiser = denoiser
        print("Loaded pretrained denoiser.")


def get_obs_for_date(df, avail_features, date_idx, env, seq_len, device):
    """Build encoder input tensors for all alive stocks at a given date."""
    date = env.dates[date_idx]
    codes = env.codes
    n = len(codes)

    dyn_list, stat_list, mask_list = [], [], []

    for i, code in enumerate(codes):
        stock_df = df[df['ts_code'] == code].sort_values('trade_date')
        date_pos = stock_df[stock_df['trade_date'] <= date]
        if len(date_pos) < seq_len:
            dyn_list.append(np.zeros((seq_len, len(avail_features)), dtype=np.float32))
            stat_list.append(np.zeros(len(STATIC_FEATURES), dtype=np.float32))
            mask_list.append(False)
            continue

        dynamic_data = date_pos[avail_features].values.astype(np.float32)
        end_idx = len(dynamic_data)
        normalized = rolling_normalize_window(dynamic_data, seq_len, end_idx)
        dyn_list.append(normalized)

        static_val = date_pos[STATIC_FEATURES].iloc[0].values.astype(np.float32)
        stat_list.append(static_val)
        mask_list.append(not env.suspended[date_idx, i])

    dyn_t = torch.tensor(np.array(dyn_list), device=device)
    stat_t = torch.tensor(np.array(stat_list), device=device)
    mask_t = torch.tensor(mask_list, device=device, dtype=torch.bool)
    return dyn_t, stat_t, mask_t


def build_port_state(env, device):
    """Build portfolio state tensor [N_stocks, 4]."""
    nav = env._compute_nav()
    n = env.n_stocks
    prices = env.close_prices[env.current_idx]

    cash_frac = np.full(n, env.cash / (nav + 1e-8), dtype=np.float32)
    hold_val = env.holdings.astype(np.float64) * np.nan_to_num(prices, 0)
    hold_frac = (hold_val / (nav + 1e-8)).astype(np.float32)
    lock_val = env.locked.astype(np.float64) * np.nan_to_num(prices, 0)
    lock_frac = (lock_val / (nav + 1e-8)).astype(np.float32)
    prev_w = env.prev_weights.astype(np.float32)

    state = np.stack([cash_frac, hold_frac, lock_frac, prev_w], axis=-1)
    return torch.tensor(state, device=device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df, avail_features = prepare_data()
    env = build_env(df)

    dynamic_dim = len(avail_features)
    static_dim = len(STATIC_FEATURES)

    encoder = TFTEncoder(dynamic_dim, static_dim,
                         hidden_dim=config.HIDDEN_DIM,
                         seq_len=config.SEQ_LEN,
                         num_heads=config.NUM_HEADS,
                         dropout=config.DROPOUT).to(device)
    load_pretrained_encoder(encoder, device)

    policy = PortfolioPolicy(config.HIDDEN_DIM, n_bins=config.N_BINS,
                             n_extra_state=4, dropout=config.DROPOUT).to(device)

    trainer = GRPOTrainer(encoder, policy, env, device=device)

    train_dates = [i for i, d in enumerate(env.dates)
                   if config.TRAIN_START <= d <= config.TRAIN_END]
    seq_len = config.SEQ_LEN

    print(f"RL training for {config.RL_STEPS} steps...")
    best_reward = -np.inf

    for step in range(config.RL_STEPS):
        date_idx = random.choice(train_dates[seq_len:])
        phase = random.choice(["open", "close"])

        env.reset(start_date_idx=date_idx)
        env.phase = phase

        dyn_t, stat_t, mask_t = get_obs_for_date(
            df, avail_features, date_idx, env, seq_len, device)
        port_state = build_port_state(env, device)

        metrics = trainer.collect_and_update(
            dyn_t, stat_t, port_state, mask_t, phase)

        if step % 50 == 0:
            print(f"Step {step}/{config.RL_STEPS} | "
                  f"loss={metrics['loss']:.4f} | "
                  f"reward={metrics['mean_reward']:.6f} | "
                  f"kl={metrics['kl']:.4f}")

        if metrics['mean_reward'] > best_reward:
            best_reward = metrics['mean_reward']
            save_path = os.path.join(config.CACHE_DIR, "best_rl_policy.pt")
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            torch.save({
                "encoder": encoder.state_dict(),
                "policy": policy.state_dict(),
                "step": step,
            }, save_path)

    print(f"Training done. Best reward: {best_reward:.6f}")
    print(f"Model saved to {os.path.join(config.CACHE_DIR, 'best_rl_policy.pt')}")


if __name__ == "__main__":
    main()
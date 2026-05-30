import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data_loader import build_merged_dataset, universe_tag
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from feature_engine import STATIC_CATEGORICAL, STATIC_CONTINUOUS
from model import TFTEncoder, PortfolioPolicy, DiffusionDenoiser, ReturnPredictor
from dataset import StockDataset
from env import AShareTradingEnv
from grpo_trainer import GRPOTrainer
from rl_utils import build_port_state, ObsCache
from plot import plot_rl_reward_curve


SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data():
    print("Loading and preparing data...")
    df = build_merged_dataset(config.TRAIN_START, config.TEST_END)
    df, avail_features = build_features(df)
    return df, avail_features


def build_env(df):
    env = AShareTradingEnv(df)
    return env


def load_denoiser(device, dynamic_dim):
    """Load pretrained denoiser and freeze it."""
    if not config.USE_DIFFUSION_DENOISER:
        return None
    ckpt_path = os.path.join(config.CACHE_DIR, "denoiser_pretrained.pt")
    if not os.path.exists(ckpt_path):
        print(f"WARNING: {ckpt_path} not found. Running without denoiser.")
        return None
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    denoiser = DiffusionDenoiser(
        feature_dim=dynamic_dim,
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
    print("Loaded and froze pretrained denoiser.")
    return denoiser


def warmup_encoder(encoder, return_predictor, df, avail_features, device):
    """方案C：用收益预测目标对encoder做几个epoch预热。"""
    print("Warming up encoder with return prediction...")
    train_df = df[(df['trade_date'] >= config.TRAIN_START) &
                  (df['trade_date'] <= config.TRAIN_END)]
    dataset = StockDataset(train_df, avail_features, STATIC_FEATURES,
                           config.SEQ_LEN, config.PRED_HORIZON)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                        shuffle=True, num_workers=0)
    encoder.train()
    return_predictor.train()
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(return_predictor.parameters()),
        lr=config.WARMUP_LR)
    for epoch in range(config.WARMUP_EPOCHS):
        total_loss = 0.0
        n_batch = 0
        for x_dyn, x_stat, y in loader:
            x_dyn = x_dyn.to(device)
            x_stat = x_stat.to(device)
            y = y.to(device).float()
            if y.dim() > 1:
                y = y[:, 0]
            enc_out = encoder(x_dyn, x_stat)
            pred = return_predictor(enc_out)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch += 1
        print(f"  Warmup epoch {epoch+1}/{config.WARMUP_EPOCHS} "
              f"loss={total_loss/max(n_batch,1):.6f}")
    print("Encoder warmup complete.")


def randomize_portfolio_state(env, date_idx):
    """Randomly initialize portfolio state so policy trains on diverse conditions."""
    prices = env.close_prices[date_idx]
    valid = ~np.isnan(prices) & (prices > 0)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0 or random.random() < 0.2:
        return

    n_held = random.randint(1, min(config.N_HOLD, len(valid_indices)))
    held_indices = np.random.choice(valid_indices, size=n_held, replace=False)

    cash_frac = random.uniform(0.1, 0.6)
    invest_capital = env.init_capital * (1 - cash_frac)
    env.cash = env.init_capital * cash_frac

    weights = np.random.dirichlet(np.ones(n_held))
    for j, idx in enumerate(held_indices):
        shares = int(weights[j] * invest_capital / prices[idx]
                     / config.LOT) * config.LOT
        if random.random() < 0.3:
            env.locked[idx] = shares
        else:
            env.holdings[idx] = shares

    nav = env._compute_nav(prices)
    if nav > 0:
        total_held = (env.holdings + env.locked).astype(np.float64)
        env.prev_weights = (total_held * prices) / nav
        env.prev_weights = np.nan_to_num(env.prev_weights, 0.0)


def main():
    set_seed(SEED)
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
                         dropout=config.DROPOUT,
                         static_categorical=STATIC_CATEGORICAL,
                         static_n_continuous=len(STATIC_CONTINUOUS)).to(device)

    denoiser = load_denoiser(device, dynamic_dim)
    if denoiser is not None:
        encoder.denoiser = denoiser

    return_predictor = ReturnPredictor(config.HIDDEN_DIM).to(device)

    warmup_encoder(encoder, return_predictor, df, avail_features, device)

    policy = PortfolioPolicy(config.HIDDEN_DIM, n_bins=config.N_BINS,
                             n_extra_state=config.N_EXTRA_STATE,
                             dropout=config.DROPOUT).to(device)

    trainer = GRPOTrainer(encoder, policy, env,
                          return_predictor=return_predictor, device=device)

    train_dates = [i for i, d in enumerate(env.dates)
                   if config.TRAIN_START <= d <= config.TRAIN_END]
    seq_len = config.SEQ_LEN

    valid_starts = [d for d in train_dates[seq_len:]
                    if d + config.EPISODE_LEN - 1 < len(env.dates)]

    grouped = df.sort_values('trade_date').groupby('ts_code')
    obs_cache = ObsCache(grouped, avail_features, env, seq_len)

    print(f"RL training for {config.RL_STEPS} episodes "
          f"({len(valid_starts)} valid starts)...")
    print("Training encoder + policy end-to-end with GRPO from scratch.")
    best_reward = -np.inf
    reward_history = []
    t_start = time.time()

    for step in range(config.RL_STEPS):
        start_idx = random.choice(valid_starts)

        env.reset(start_date_idx=start_idx, episode_len=config.EPISODE_LEN)
        randomize_portfolio_state(env, start_idx)

        metrics = trainer.collect_trajectory_and_update(
            env, obs_cache, start_idx, device)

        reward_history.append(metrics['mean_reward'])
        if step % 10 == 0:
            elapsed = time.time() - t_start
            speed = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (config.RL_STEPS - step - 1) / speed if speed > 0 else 0
            print(f"Episode {step}/{config.RL_STEPS} | "
                  f"loss={metrics['loss']:.4f} | "
                  f"mean_reward={metrics['mean_reward']:.6f} | "
                  f"best_reward={metrics['best_reward']:.6f} | "
                  f"kl={metrics['kl']:.4f} | "
                  f"{speed:.2f} ep/s | ETA {eta/60:.1f}min")

        window = reward_history[-100:]
        avg_reward = np.mean(window)
        if len(reward_history) >= 100 and avg_reward > best_reward:
            best_reward = avg_reward
            save_path = os.path.join(config.CACHE_DIR, "best_rl_policy.pt")
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            torch.save({
                "encoder": encoder.state_dict(),
                "policy": policy.state_dict(),
                "step": step,
                "config": {
                    'UNIVERSE': universe_tag(getattr(config, 'UNIVERSE', None)),
                    'HIDDEN_DIM': config.HIDDEN_DIM,
                    'NUM_HEADS': config.NUM_HEADS,
                    'SEQ_LEN': config.SEQ_LEN,
                    'DROPOUT': config.DROPOUT,
                    'N_BINS': config.N_BINS,
                    'BINS': config.BINS,
                    'N_HOLD': config.N_HOLD,
                    'N_EXTRA_STATE': config.N_EXTRA_STATE,
                    'EPISODE_LEN': config.EPISODE_LEN,
                    'MAX_CASH': config.MAX_CASH,
                    'USE_DIFFUSION_DENOISER': config.USE_DIFFUSION_DENOISER,
                    'DYNAMIC_FEATURES': avail_features,
                },
            }, save_path)

    print(f"Training done. Best avg reward (100-step): {best_reward:.6f}")
    total_time = time.time() - t_start
    print(f"Total training time: {total_time/60:.1f} min "
          f"({total_time/config.RL_STEPS:.2f} s/episode)")
    print(f"Model saved to {os.path.join(config.CACHE_DIR, 'best_rl_policy.pt')}")
    plot_rl_reward_curve(reward_history)


if __name__ == "__main__":
    main()
